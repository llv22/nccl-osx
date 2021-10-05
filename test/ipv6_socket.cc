/**
 * For fixing the issue of https://github.com/llv22/nccl-osx/issues/2.
 */
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <netdb.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <errno.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/syslimits.h>
#include <algorithm>
#include <map>
#include <string>

#define MAX_IFS 16
#define MAX_IF_NAME_SIZE 16
#define SLEEP_INT 1000          // connection retry sleep interval in usec
#define RETRY_REFUSED_TIMES 2e4 // connection refused retry times before reporting a timeout (20 sec)
#define RETRY_TIMEDOUT_TIMES 3  // connection timed out retry times (each one can take 20s)
#define INFO(...) printf(__VA_ARGS__)
#define WARN(...) fprintf(stderr, __VA_ARGS__)

#define NCCL_UNIQUE_ID_BYTES 128
typedef struct
{
  char internal[NCCL_UNIQUE_ID_BYTES];
} ncclUniqueId;
#define NCCL_NET_HANDLE_MAXSIZE 64
typedef char ncclNetHandle_t[NCCL_NET_HANDLE_MAXSIZE];

struct extInfo
{
  int rank;
  int nranks;
  ncclNetHandle_t extHandleListenRoot;
  ncclNetHandle_t extHandleListen;
};

/* Socket Interface Selection type */
enum bootstrapInterface_t
{
  findSubnetIf = -1,
  dontCareIf = -2
};

struct netIf
{
  char prefix[64];
  int port;
};

struct linkedAddr
{
  bool supportIPv4;
  bool supportIPv6;
};

/* Common socket address storage structure for IPv4/IPv6 */
union socketAddress
{
  struct sockaddr sa;
  struct sockaddr_in sin;
  struct sockaddr_in6 sin6;
};

struct bootstrapNetComm
{
  int fd;
};

/* Init functions */
static char ncclNetIfNames[MAX_IF_NAME_SIZE * MAX_IFS];
static union socketAddress ncclNetIfAddrs[MAX_IFS];
static int ncclNetIfs = -1;
pthread_mutex_t ncclSocketLock = PTHREAD_MUTEX_INITIALIZER;

static bool matchPort(const int port1, const int port2)
{
  if (port1 == -1)
    return true;
  if (port2 == -1)
    return true;
  if (port1 == port2)
    return true;
  return false;
}

static bool matchIf(const char *string, const char *ref, bool matchExact)
{
  // Make sure to include '\0' in the exact case
  int matchLen = matchExact ? strlen(string) + 1 : strlen(ref);
  return strncmp(string, ref, matchLen) == 0;
}

int parseStringList(const char *string, struct netIf *ifList, int maxList)
{
  if (!string)
    return 0;

  const char *ptr = string;

  int ifNum = 0;
  int ifC = 0;
  char c;
  do
  {
    c = *ptr;
    if (c == ':')
    {
      if (ifC > 0)
      {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = atoi(ptr + 1);
        ifNum++;
        ifC = 0;
      }
      while (c != ',' && c != '\0')
        c = *(++ptr);
    }
    else if (c == ',' || c == '\0')
    {
      if (ifC > 0)
      {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = -1;
        ifNum++;
        ifC = 0;
      }
    }
    else
    {
      ifList[ifNum].prefix[ifC] = c;
      ifC++;
    }
    ptr++;
  } while (ifNum < maxList && c);
  return ifNum;
}

/* Allow the user to force the IPv4/IPv6 interface selection */
static inline int envSocketFamily(void)
{
  int family = -1; // Family selection is not forced, will use first one found
  char *env = getenv("NCCL_SOCKET_FAMILY");
  if (env == NULL)
  {
    return family;
  }

  if (strcmp(env, "AF_INET") == 0)
    family = AF_INET; // IPv4
  else if (strcmp(env, "AF_INET6") == 0)
    family = AF_INET6; // IPv6
  return family;
}

static bool matchSubnet(struct ifaddrs local_if, union socketAddress *remote)
{
  /* Check family first */
  int family = local_if.ifa_addr->sa_family;
  if (family != remote->sa.sa_family)
  {
    return false;
  }

  if (family == AF_INET)
  {
    struct sockaddr_in *local_addr = (struct sockaddr_in *)(local_if.ifa_addr);
    struct sockaddr_in *mask = (struct sockaddr_in *)(local_if.ifa_netmask);
    struct sockaddr_in &remote_addr = remote->sin;
    struct in_addr local_subnet, remote_subnet;
    local_subnet.s_addr = local_addr->sin_addr.s_addr & mask->sin_addr.s_addr;
    remote_subnet.s_addr = remote_addr.sin_addr.s_addr & mask->sin_addr.s_addr;
    return (local_subnet.s_addr ^ remote_subnet.s_addr) ? false : true;
  }
  else if (family == AF_INET6)
  {
    struct sockaddr_in6 *local_addr = (struct sockaddr_in6 *)(local_if.ifa_addr);
    struct sockaddr_in6 *mask = (struct sockaddr_in6 *)(local_if.ifa_netmask);
    struct sockaddr_in6 &remote_addr = remote->sin6;
    struct in6_addr &local_in6 = local_addr->sin6_addr;
    struct in6_addr &mask_in6 = mask->sin6_addr;
    struct in6_addr &remote_in6 = remote_addr.sin6_addr;
    bool same = true;
    int len = 16; //IPv6 address is 16 unsigned char
    for (int c = 0; c < len; c++)
    { //Network byte order is big-endian
      char c1 = local_in6.s6_addr[c] & mask_in6.s6_addr[c];
      char c2 = remote_in6.s6_addr[c] & mask_in6.s6_addr[c];
      if (c1 ^ c2)
      {
        same = false;
        break;
      }
    }
    // At last, we need to compare scope id
    // Two Link-type addresses can have the same subnet address even though they are not in the same scope
    // For Global type, this field is 0, so a comparison wouldn't matter
    same &= (local_addr->sin6_scope_id == remote_addr.sin6_scope_id);
    return same;
  }
  else
  {
    WARN("Net : Unsupported address family type");
    return false;
  }
}

/* Format a string representation of a (struct sockaddr *) socket address using getnameinfo()
 *
 * Output: "IPv4/IPv6 address<port>"
 */
static inline const char *socketToString(struct sockaddr *saddr, char *buf)
{
  if (buf == NULL || saddr == NULL)
    return NULL;
  if (saddr->sa_family != AF_INET && saddr->sa_family != AF_INET6)
  {
    buf[0] = '\0';
    return buf;
  }
  char host[NI_MAXHOST], service[NI_MAXSERV];
  (void)getnameinfo(saddr, sizeof(union socketAddress), host, NI_MAXHOST, service, NI_MAXSERV, NI_NUMERICHOST | NI_NUMERICSERV);
  sprintf(buf, "%s<%s>", host, service);
  return buf;
}

bool matchIfList(const char *string, int port, struct netIf *ifList, int listSize, bool matchExact)
{
  // Make an exception for the case where no user list is defined
  if (listSize == 0)
    return true;

  for (int i = 0; i < listSize; i++)
  {
    if (matchIf(string, ifList[i].prefix, matchExact) && matchPort(port, ifList[i].port))
    {
      return true;
    }
  }
  return false;
}

static int GetSocketAddrFromString(union socketAddress *ua, const char *ip_port_pair)
{
  if (!(ip_port_pair && strlen(ip_port_pair) > 1))
  {
    WARN("Net : string is null");
    return -1;
  }

  bool ipv6 = ip_port_pair[0] == '[';
  /* Construct the sockaddress structure */
  if (!ipv6)
  {
    struct netIf ni;
    // parse <ip_or_hostname>:<port> string, expect one pair
    if (parseStringList(ip_port_pair, &ni, 1) != 1)
    {
      WARN("Net : No valid <IPv4_or_hostname>:<port> pair found");
      return -1;
    }

    struct addrinfo hints, *p;
    int rv;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if ((rv = getaddrinfo(ni.prefix, NULL, &hints, &p)) != 0)
    {
      WARN("Net : error encountered when getting address info : %s", gai_strerror(rv));
      return -1;
    }

    // use the first
    if (p->ai_family == AF_INET)
    {
      struct sockaddr_in &sin = ua->sin;
      memcpy(&sin, p->ai_addr, sizeof(struct sockaddr_in));
      sin.sin_family = AF_INET; // IPv4
      //inet_pton(AF_INET, ni.prefix, &(sin.sin_addr));  // IP address
      sin.sin_port = htons(ni.port); // port
    }
    else if (p->ai_family == AF_INET6)
    {
      struct sockaddr_in6 &sin6 = ua->sin6;
      memcpy(&sin6, p->ai_addr, sizeof(struct sockaddr_in6));
      sin6.sin6_family = AF_INET6;     // IPv6
      sin6.sin6_port = htons(ni.port); // port
      sin6.sin6_flowinfo = 0;          // needed by IPv6, but possibly obsolete
      sin6.sin6_scope_id = 0;          // should be global scope, set to 0
    }
    else
    {
      WARN("Net : unsupported IP family");
      return -1;
    }

    freeaddrinfo(p); // all done with this structure
  }
  else
  {
    int i, j = -1, len = strlen(ip_port_pair);
    for (i = 1; i < len; i++)
    {
      if (ip_port_pair[i] == '%')
        j = i;
      if (ip_port_pair[i] == ']')
        break;
    }
    if (i == len)
    {
      WARN("Net : No valid [IPv6]:port pair found");
      return -1;
    }
    bool global_scope = (j == -1 ? true : false); // If no % found, global scope; otherwise, link scope

    char ip_str[NI_MAXHOST], port_str[NI_MAXSERV], if_name[IFNAMSIZ];
    memset(ip_str, '\0', sizeof(ip_str));
    memset(port_str, '\0', sizeof(port_str));
    memset(if_name, '\0', sizeof(if_name));
    strncpy(ip_str, ip_port_pair + 1, global_scope ? i - 1 : j - 1);
    strncpy(port_str, ip_port_pair + i + 2, len - i - 1);
    int port = atoi(port_str);
    if (!global_scope)
      strncpy(if_name, ip_port_pair + j + 1, i - j - 1); // If not global scope, we need the intf name

    struct sockaddr_in6 &sin6 = ua->sin6;
    sin6.sin6_family = AF_INET6;                                     // IPv6
    inet_pton(AF_INET6, ip_str, &(sin6.sin6_addr));                  // IP address
    sin6.sin6_port = htons(port);                                    // port
    sin6.sin6_flowinfo = 0;                                          // needed by IPv6, but possibly obsolete
    sin6.sin6_scope_id = global_scope ? 0 : if_nametoindex(if_name); // 0 if global scope; intf index if link scope
  }
  return 0;
}

static int findInterfaces(const char *prefixList, char *names, union socketAddress *addrs, int sock_family, int maxIfNameSize, int maxIfs)
{
#ifdef ENABLE_TRACE
  char line[1024];
#endif
  struct netIf userIfs[MAX_IFS];
  bool searchNot = prefixList && prefixList[0] == '^';
  if (searchNot)
    prefixList++;
  bool searchExact = prefixList && prefixList[0] == '=';
  if (searchExact)
    prefixList++;
  int nUserIfs = parseStringList(prefixList, userIfs, MAX_IFS);
  
#if defined(__APPLE__) && defined(__MACH__)
  using namespace std;
  map<string, linkedAddr*> ethMap;
  struct ifaddrs *interfaces_p, *interface_p;
  getifaddrs(&interfaces_p);
  int c = 0;

  for (interface_p = interfaces_p; interface_p && c < maxIfs; interface_p = interface_p->ifa_next)
  {
    if (interface_p->ifa_addr == NULL)
      continue;
    /* We only support IPv4 & IPv6 */
    int family = interface_p->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6)
      continue;
    if ((interface_p->ifa_flags & (IFF_UP|IFF_RUNNING|IFF_LOOPBACK)) != (IFF_UP|IFF_RUNNING)) {
      //see: if not up and running
      continue;
    }

    string ethName = interface_p->ifa_name;
    map<string, linkedAddr*>::iterator it = ethMap.find(ethName);
    if (it != ethMap.end()) {
      if (family == AF_INET6) {
        it->second->supportIPv6 = true;
      }
      if (family == AF_INET) {
        it->second->supportIPv4 = true;
      }
    }
    else {
      linkedAddr* addr = new linkedAddr; 
      if (family == AF_INET6) {
        addr->supportIPv6 = true;
      }
      if (family == AF_INET) {
        addr->supportIPv4 = true;
      }
      ethMap[ethName] = addr;
    }
    /* We also need to skip IPv6 loopback interfaces */
    if (family == AF_INET6)
    {
      struct sockaddr_in6 *sa = (struct sockaddr_in6 *)(interface_p->ifa_addr);
      if (IN6_IS_ADDR_LOOPBACK(&sa->sin6_addr))
        continue;
    }

    // check against user specified interfaces
    if (!(matchIfList(interface_p->ifa_name, -1, userIfs, nUserIfs, searchExact) ^ searchNot))
    {
      continue;
    }

    // Check that this interface has not already been saved
    // getifaddrs() normal order appears to be; IPv4, IPv6 Global, IPv6 Link
    bool duplicate = false;
    for (int i = 0; i < c; i++)
    {
      if (strcmp(interface_p->ifa_name, names + i * maxIfNameSize) == 0)
      {
        duplicate = true;
        break;
      }
    }

    if (!duplicate)
    {
      c++;
    }
  }

  freeifaddrs(interfaces_p);
#endif

  int found = 0;
  struct ifaddrs *interfaces, *interface;
  getifaddrs(&interfaces);
  for (interface = interfaces; interface && found < maxIfs; interface = interface->ifa_next)
  {
    if (interface->ifa_addr == NULL)
      continue;

    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6)
      continue;
    if ((interface->ifa_flags & (IFF_UP|IFF_RUNNING|IFF_LOOPBACK)) != (IFF_UP|IFF_RUNNING)) {
      //see: if not up and running
      continue;
    }

    char line[1024];
    line[0] = '\0';
    socketToString(interface->ifa_addr, line);
    line[1023] = '\0';
    printf("Found interface %s:%s\n", interface->ifa_name, line);

    /* Allow the caller to force the socket family type */
    if (sock_family != -1 && family != sock_family)
      continue;

    /* We also need to skip IPv6 loopback interfaces */
    if (family == AF_INET6)
    {
      struct sockaddr_in6 *sa = (struct sockaddr_in6 *)(interface->ifa_addr);
      if (IN6_IS_ADDR_LOOPBACK(&sa->sin6_addr))
        continue;
    }

    // check against user specified interfaces
    if (!(matchIfList(interface->ifa_name, -1, userIfs, nUserIfs, searchExact) ^ searchNot))
    {
      continue;
    }

    // Check that this interface has not already been saved
    // getifaddrs() normal order appears to be; IPv4, IPv6 Global, IPv6 Link
    bool duplicate = false;
    for (int i = 0; i < found; i++)
    {
      if (strcmp(interface->ifa_name, names + i * maxIfNameSize) == 0)
      {
        duplicate = true;
        break;
      }
    }

    if (!duplicate)
    {
  #if defined(__APPLE__) && defined(__MACH__)
      string ethName = interface->ifa_name;
      map<string, linkedAddr*>::iterator it = ethMap.find(ethName);
      if (it != ethMap.end() && it->second->supportIPv4 && it->second->supportIPv6) {
        // Store the interface name
        strncpy(names + found * maxIfNameSize, interface->ifa_name, maxIfNameSize);
        // Store the IP address
        int salen = (family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);
        memcpy(addrs + found, interface->ifa_addr, salen);
        found++;
      }
  #else
      // Store the interface name
      strncpy(names + found * maxIfNameSize, interface->ifa_name, maxIfNameSize);
      // Store the IP address
      int salen = (family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);
      memcpy(addrs + found, interface->ifa_addr, salen);
      found++;
  #endif
    }
  }

  freeifaddrs(interfaces);

  return found;
}

static int findInterfaceMatchSubnet(char *ifNames, union socketAddress *localAddrs, union socketAddress *remoteAddr, int ifNameMaxSize, int maxIfs)
{
#ifdef ENABLE_TRACE
  char line[1024];
#endif
  char line_a[1024];
  int found = 0;
  struct ifaddrs *interfaces, *interface;
  getifaddrs(&interfaces);
  for (interface = interfaces; interface && !found; interface = interface->ifa_next)
  {
    if (interface->ifa_addr == NULL)
      continue;

    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6)
      continue;

    // check against user specified interfaces
    if (!matchSubnet(*interface, remoteAddr))
    {
      continue;
    }

    // Store the local IP address
    int salen = (family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);
    memcpy(localAddrs + found, interface->ifa_addr, salen);

    // Store the interface name
    strncpy(ifNames + found * ifNameMaxSize, interface->ifa_name, ifNameMaxSize);

    // TRACE(NCCL_INIT|NCCL_NET,"NET : Found interface %s:%s in the same subnet as remote address %s", interface->ifa_name, socketToString(&(localAddrs[found].sa), line), socketToString(&(remoteAddr->sa), line_a));
    found++;
    if (found == maxIfs)
      break;
  }

  if (found == 0)
  {
    WARN("Net : No interface found in the same subnet as remote address %s", socketToString(&(remoteAddr->sa), line_a));
  }
  freeifaddrs(interfaces);
  return found;
}

static int findInterfaces(char *ifNames, union socketAddress *ifAddrs, int ifNameMaxSize, int maxIfs)
{
  int nIfs = 0;
  // Allow user to force the INET socket family selection
  int sock_family = envSocketFamily();
  // User specified interface
  char *env = getenv("NCCL_SOCKET_IFNAME");
  if (env && strlen(env) > 1)
  {
    // Specified by user : find or fail
    nIfs = findInterfaces(env, ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
  }
  else
  {
    // Try to automatically pick the right one
    // Start with IB
    nIfs = findInterfaces("ib", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
    // else see if we can get some hint from COMM ID
    if (nIfs == 0)
    {
      char *commId = getenv("NCCL_COMM_ID");
      if (commId && strlen(commId) > 1)
      {
        // Try to find interface that is in the same subnet as the IP in comm id
        union socketAddress idAddr;
        GetSocketAddrFromString(&idAddr, commId);
        nIfs = findInterfaceMatchSubnet(ifNames, ifAddrs, &idAddr, ifNameMaxSize, maxIfs);
      }
    }
    // Then look for anything else (but not docker or lo)
    if (nIfs == 0)
      nIfs = findInterfaces("^docker,lo", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
    // Finally look for docker, then lo.
    if (nIfs == 0)
      nIfs = findInterfaces("docker", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
    if (nIfs == 0)
      nIfs = findInterfaces("lo", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
  }
  return nIfs;
}

int ncclSocketInit()
{
  if (ncclNetIfs == -1)
  {
    pthread_mutex_lock(&ncclSocketLock);
    if (ncclNetIfs == -1)
    {
      ncclNetIfs = findInterfaces(ncclNetIfNames, ncclNetIfAddrs, MAX_IF_NAME_SIZE, MAX_IFS);
      if (ncclNetIfs <= 0)
      {
        WARN("NET/Socket : no interface found");
        return -1;
      }
      else
      {
        char line[1024];
        char addrline[1024];
        line[0] = '\0';
        for (int i = 0; i < ncclNetIfs; i++)
        {
          snprintf(line + strlen(line), 1023 - strlen(line), " [%d]%s:%s", i, ncclNetIfNames + i * MAX_IF_NAME_SIZE,
                   socketToString(&ncclNetIfAddrs[i].sa, addrline));
        }
        line[1023] = '\0';
        INFO("NET/Socket : Using%s\n", line);
      }
    }
    pthread_mutex_unlock(&ncclSocketLock);
  }
  return 0;
}

static inline uint16_t socketToPort(struct sockaddr *saddr)
{
  return ntohs(saddr->sa_family == AF_INET ? ((struct sockaddr_in *)saddr)->sin_port : ((struct sockaddr_in6 *)saddr)->sin6_port);
}

static int createListenSocket(int *fd, union socketAddress *localAddr)
{
  /* IPv4/IPv6 support */
  int family = localAddr->sa.sa_family;
  int salen = (family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);

  /* Create socket and bind it to a port */
  int sockfd = socket(family, SOCK_STREAM, 0);
  if (sockfd == -1)
  {
    WARN("Net : Socket creation failed : %s", strerror(errno));
    return -1;
  }

  if (socketToPort(&localAddr->sa))
  {
    // Port is forced by env. Make sure we get the port.
    int opt = 1;
#if defined(SO_REUSEPORT)
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));
#else
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif
  }

  // localAddr port should be 0 (Any port)
  bind(sockfd, &localAddr->sa, salen);

  /* Get the assigned Port */
  socklen_t size = salen;
  getsockname(sockfd, &localAddr->sa, &size);

  char line[1024];
  line[0] = '\0';
  socketToString(&localAddr->sa, line);
  line[1023] = '\0';
  INFO("Listening on socket %s\n", line);

  /* Put the socket in listen mode
   * NB: The backlog will be silently truncated to the value in /proc/sys/net/core/somaxconn
   */
  listen(sockfd, 16384);
  *fd = sockfd;
  return 0;
}

static int connectAddress(int *fd, union socketAddress *remoteAddr)
{
  /* IPv4/IPv6 support */
  int family = remoteAddr->sa.sa_family;
  int salen = (family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);

  /* Connect to a hostname / port */
  *fd = socket(family, SOCK_STREAM, 0);
  if (*fd == -1)
  {
    WARN("Net : Socket creation failed : %s", strerror(errno));
    return -1;
  }

  const int one = 1;
  setsockopt(*fd, IPPROTO_TCP, TCP_NODELAY, (char *)&one, sizeof(int));
  // SYSCHECK(setsockopt(*fd, SOL_SOCKET, SO_REUSEADDR , (char*)&one, sizeof(int)), "setsockopt");

  /*  const int bufsize = 128*1024;
    SYSCHECK(setsockopt(*fd, SOL_SOCKET, SO_SNDBUF, (char*)&bufsize, sizeof(int)), "setsockopt");
    SYSCHECK(setsockopt(*fd, SOL_SOCKET, SO_RCVBUF, (char*)&bufsize, sizeof(int)), "setsockopt");*/

  char line[1024];
  // #ifdef ENABLE_TRACE
  INFO("Connecting to socket %s", socketToString(&remoteAddr->sa, line));
  // TRACE(NCCL_INIT|NCCL_NET,"Connecting to socket %s", socketToString(&remoteAddr->sa, line));
  // #endif

  int ret;
  int timedout_retries = 0;
  int refused_retries = 0;
retry:
  connect(*fd, (struct sockaddr *)&remoteAddr->sin6, salen);
  // connect(*fd, &remoteAddr->sa, salen);
  if (ret == 0)
    return 0;
  if ((errno == ECONNREFUSED || errno == ETIMEDOUT))
  {
    if ((errno == ECONNREFUSED && ++refused_retries < RETRY_REFUSED_TIMES) ||
        (errno == ETIMEDOUT && ++timedout_retries < RETRY_TIMEDOUT_TIMES))
    {
      if (refused_retries % 1000 == 0)
      {
        INFO("Call to connect returned %s, retrying", strerror(errno));
      }
      usleep(SLEEP_INT);
      goto retry;
    }
  }
  WARN("Connect to %s failed : %s", socketToString(&remoteAddr->sa, line), strerror(errno));
  return -1;
}

#define NCCL_SOCKET_SEND 0
#define NCCL_SOCKET_RECV 1
static int socketProgressOpt(int op, int fd, void *ptr, int size, int *offset, int block)
{
  int bytes = 0;
  char *data = (char *)ptr;
  do
  {
    if (op == NCCL_SOCKET_RECV)
      bytes = recv(fd, data + (*offset), size - (*offset), block ? 0 : MSG_DONTWAIT);
    if (op == NCCL_SOCKET_SEND)
      bytes = send(fd, data + (*offset), size - (*offset), block ? 0 : MSG_DONTWAIT);
    if (op == NCCL_SOCKET_RECV && bytes == 0)
    {
      WARN("Net : Connection closed by remote peer");
      return -1;
    }
    if (bytes == -1)
    {
      if (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN)
      {
        WARN("Call to recv failed : %s", strerror(errno));
        return -1;
      }
      else
      {
        bytes = 0;
      }
    }
    (*offset) += bytes;
  } while (bytes > 0 && (*offset) < size);
  return 0;
}

/* Init functions */
static char bootstrapNetIfNames[MAX_IF_NAME_SIZE * MAX_IFS];
static union socketAddress bootstrapNetIfAddrs[MAX_IFS];
static int bootstrapNetIfs = -1;
pthread_mutex_t bootstrapNetLock = PTHREAD_MUTEX_INITIALIZER;

int bootstrapNetInit()
{
  if (bootstrapNetIfs == -1)
  {
    pthread_mutex_lock(&bootstrapNetLock);
    if (bootstrapNetIfs == -1)
    {
      bootstrapNetIfs = findInterfaces(bootstrapNetIfNames, bootstrapNetIfAddrs, MAX_IF_NAME_SIZE, MAX_IFS);
      if (bootstrapNetIfs <= 0)
      {
        WARN("Bootstrap : no socket interface found");
        return -1;
      }
      else
      {
        char line[1024];
        char addrline[1024];
        line[0] = '\0';
        for (int i = 0; i < bootstrapNetIfs; i++)
        {
          snprintf(line + strlen(line), 1023 - strlen(line), " [%d]%s:%s", i, bootstrapNetIfNames + i * MAX_IF_NAME_SIZE,
                   socketToString(&bootstrapNetIfAddrs[i].sa, addrline));
        }
        line[1023] = '\0';
        INFO("Bootstrap : Using%s\n", line);
      }
    }
    pthread_mutex_unlock(&bootstrapNetLock);
  }
  return 0;
}

static int bootstrapNetGetSocketAddr(int dev, union socketAddress *addr)
{
  if (dev >= bootstrapNetIfs)
    return -1;
  memcpy(addr, bootstrapNetIfAddrs + dev, sizeof(*addr));
  return 0;
}

template <typename T>
static int ncclCalloc(T **ptr, size_t nelem)
{
  void *p = malloc(nelem * sizeof(T));
  if (p == NULL)
  {
    WARN("Failed to malloc %ld bytes", nelem * sizeof(T));
    return -1;
  }
  memset(p, 0, nelem * sizeof(T));
  *ptr = (T *)p;
  return 0;
}

static int bootstrapNetNewComm(struct bootstrapNetComm **comm)
{
  ncclCalloc(comm, 1);
  (*comm)->fd = -1;
  return 0;
}

static int setFilesLimit()
{
  struct rlimit filesLimit;
  getrlimit(RLIMIT_NOFILE, &filesLimit);
  filesLimit.rlim_cur = filesLimit.rlim_max;
  // see: orlando already fixed this issue, "setrlimit cause warning on osx: [0] bootstrap.cc:162 NCCL WARN Call to setrlimit failed : Invalid argument"
  // refer to https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/setrlimit.2.html
  if (filesLimit.rlim_cur > OPEN_MAX)
  {
    filesLimit.rlim_cur = OPEN_MAX;
    INFO("adjust filesLimit.rlim_cur to OPEN_MAX = %lld\n", filesLimit.rlim_cur);
  }
  setrlimit(RLIMIT_NOFILE, &filesLimit);
  return 0;
}

static int bootstrapNetAccept(void *listenComm, void **recvComm)
{
  struct bootstrapNetComm *lComm = (struct bootstrapNetComm *)listenComm;
  struct bootstrapNetComm *rComm;
  bootstrapNetNewComm(&rComm);
  struct sockaddr_in6 sockaddr;
  socklen_t socklen = sizeof(sockaddr);
  accept(lComm->fd, (struct sockaddr *)&sockaddr, &socklen);
  INFO("accept %d:%d\n", sockaddr.sin6_scope_id, sockaddr.sin6_port);
  // INFO("accept %s:%d\n", sockaddr.sin_zero, sockaddr.sin_port);
  *recvComm = rComm;
  return 0;
}

static int socketWait(int op, int fd, void *ptr, int size, int *offset)
{
  while (*offset < size)
    socketProgressOpt(op, fd, ptr, size, offset, 1);
  return 0;
}

static int socketReceive(int fd, void *ptr, int size)
{
  int offset = 0;
  socketWait(NCCL_SOCKET_RECV, fd, ptr, size, &offset);
  return 0;
}

static int bootstrapNetRecv(void *recvComm, void *data, int size)
{
  struct bootstrapNetComm *comm = (struct bootstrapNetComm *)recvComm;
  int recvSize;
  socketReceive(comm->fd, &recvSize, sizeof(int));
  if (recvSize > size)
  {
    WARN("Message truncated : received %d bytes instead of %d\n", recvSize, size);
    return -1;
  }
  socketReceive(comm->fd, data, std::min(recvSize, size));
  return -1;
}

static int bootstrapNetClose(void *opaqueComm)
{
  struct bootstrapNetComm *comm = (struct bootstrapNetComm *)opaqueComm;
  if (comm)
  {
    close(comm->fd);
    free(comm);
  }
  return 0;
}

static int bootstrapNetCloseRecv(void *recvComm)
{
  bootstrapNetClose(recvComm);
  return 0;
}

static int bootstrapNetConnect(int dev, ncclNetHandle_t *netHandle, void **sendComm)
{
  union socketAddress *connectAddr = (union socketAddress *)netHandle;
  struct bootstrapNetComm *comm;
  bootstrapNetNewComm(&comm);
  connectAddress(&comm->fd, connectAddr);
  *sendComm = comm;
  return 0;
}

static int socketSend(int fd, void *ptr, int size)
{
  int offset = 0;
  socketWait(NCCL_SOCKET_SEND, fd, ptr, size, &offset);
  return 0;
}

static int bootstrapNetCloseSend(void *sendComm)
{
  bootstrapNetClose(sendComm);
  return 0;
}

static int bootstrapNetSend(void *sendComm, void *data, int size)
{
  struct bootstrapNetComm *comm = (struct bootstrapNetComm *)sendComm;
  socketSend(comm->fd, &size, sizeof(int));
  socketSend(comm->fd, data, size);
  return 0;
}

static int bootstrapNetCloseListen(void *listenComm)
{
  bootstrapNetClose(listenComm);
  return 0;
}

static void *bootstrapRoot(void *listenComm)
{
  struct extInfo info;
  ncclNetHandle_t *rankHandles = NULL;
  ncclNetHandle_t *rankHandlesRoot = NULL; // for initial rank <-> root information exchange
  ncclNetHandle_t zero = {0};              // for sanity checking
  void *tmpComm;
  int res;
  setFilesLimit();

  printf("BEGIN\n");
  /* Receive addresses from all ranks */
  int nranks = 0, c = 0;
  do
  {
    bootstrapNetAccept(listenComm, &tmpComm);
    bootstrapNetRecv(tmpComm, &info, sizeof(info));
    bootstrapNetCloseRecv(tmpComm);

    if (c == 0)
    {
      nranks = info.nranks;
      ncclCalloc(&rankHandles, nranks);
      ncclCalloc(&rankHandlesRoot, nranks);
    }

    if (nranks != info.nranks)
    {
      WARN("Bootstrap Root : mismatch in rank count from procs %d : %d", nranks, info.nranks);
      goto out;
    }

    if (memcmp(&zero, &rankHandlesRoot[info.rank], sizeof(ncclNetHandle_t)) != 0)
    {
      WARN("Bootstrap Root : rank %d of %d ranks has already checked in", info.rank, nranks);
      goto out;
    }

    // Save the connection handle for that rank
    memcpy(rankHandlesRoot + info.rank, info.extHandleListenRoot, sizeof(ncclNetHandle_t));
    memcpy(rankHandles + info.rank, info.extHandleListen, sizeof(ncclNetHandle_t));

    ++c;
    INFO("Received connect from rank %d total %d/%d\n", info.rank, c, nranks);
  } while (c < nranks);
  INFO("COLLECTED ALL %d HANDLES", nranks);

  // Send the connect handle for the next rank in the AllGather ring
  for (int r = 0; r < nranks; ++r)
  {
    int next = (r + 1) % nranks;
    void *tmpSendComm;
    bootstrapNetConnect(0, rankHandlesRoot + r, &tmpSendComm);
    bootstrapNetSend(tmpSendComm, rankHandles + next, sizeof(ncclNetHandle_t));
    bootstrapNetCloseSend(tmpSendComm);
  }
  INFO("SENT OUT ALL %d HANDLES", nranks);

out:
  bootstrapNetCloseListen(listenComm);
  if (rankHandles)
    free(rankHandles);
  if (rankHandlesRoot)
    free(rankHandlesRoot);

  INFO("DONE");
  return NULL;
}

struct ServerInfo
{
  int dev;
  ncclNetHandle_t *netHandle;
  void **listenComm;
};

static int bootstrapNetListen(int dev, ncclNetHandle_t *netHandle, void **listenComm)
{
  union socketAddress *connectAddr = (union socketAddress *)netHandle;
  static_assert(sizeof(union socketAddress) < NCCL_NET_HANDLE_MAXSIZE, "union socketAddress size is too large");
  // if dev >= 0, listen based on dev
  if (dev >= 0)
  {
    bootstrapNetGetSocketAddr(dev, connectAddr);
  }
  else if (dev == findSubnetIf)
  {
    // handle stores a remote address
    // need to find a local addr that is in the same network as the remote addr
    union socketAddress localAddr;
    char ifName[MAX_IF_NAME_SIZE];
    if (findInterfaceMatchSubnet(ifName, &localAddr, connectAddr, MAX_IF_NAME_SIZE, 1) <= 0)
    {
      WARN("NET/Socket : No usable listening interface found");
      return -1;
    }
    // pass the local address back
    memcpy(connectAddr, &localAddr, sizeof(localAddr));
  } // Otherwise, handle stores a local address
  struct bootstrapNetComm *comm;
  bootstrapNetNewComm(&comm);
  createListenSocket(&comm->fd, connectAddr);
  *listenComm = comm;
  return 0;
}

static void *bootstrapNetListenWrapper(void *c)
{
  ServerInfo *param = (ServerInfo *)c;
  bootstrapNetListen(param->dev, param->netHandle, param->listenComm);
  pthread_t client_thread;
  pthread_create(&client_thread, NULL, bootstrapRoot, param->listenComm);
  int err = pthread_join(client_thread, NULL);
  if (err != 0)
  {
    fprintf(stderr, "failed to wait thread client_thread for terminating\n");
  }
  return NULL;
}

int main()
{
  // follow the logic in bootstrap.cc::bootstrapNetListen()
  int status = bootstrapNetInit();
  printf("Socket initialization status: %d, bootstrapNetIfs: %d\n", status, bootstrapNetIfs);
  return 0;
}