﻿/**
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

#define MAX_IFS 16
#define MAX_IF_NAME_SIZE 16
#define SLEEP_INT 1000          // connection retry sleep interval in usec
#define RETRY_REFUSED_TIMES 2e4 // connection refused retry times before reporting a timeout (20 sec)
#define RETRY_TIMEDOUT_TIMES 3  // connection timed out retry times (each one can take 20s)
#define INFO(...) printf(__VA_ARGS__)
#define WARN(...) fprintf(stderr, __VA_ARGS__)

struct netIf
{
    char prefix[64];
    int port;
};

/* Common socket address storage structure for IPv4/IPv6 */
union socketAddress
{
    struct sockaddr sa;
    struct sockaddr_in sin;
    struct sockaddr_in6 sin6;
};

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

static inline uint16_t socketToPort(struct sockaddr *saddr)
{
    return ntohs(saddr->sa_family == AF_INET ? ((struct sockaddr_in *)saddr)->sin_port : ((struct sockaddr_in6 *)saddr)->sin6_port);
}

/* Allow the user to force the IPv4/IPv6 interface selection */
static inline int envSocketFamily(void)
{
    int family = -1; // Family selection is not forced, will use first one found
    char *env = getenv("NCCL_SOCKET_FAMILY");
    if (env == NULL)
        return family;

    if (strcmp(env, "AF_INET") == 0)
        family = AF_INET; // IPv4
    else if (strcmp(env, "AF_INET6") == 0)
        family = AF_INET6; // IPv6
    return family;
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

        // TRACE(NCCL_INIT|NCCL_NET,"Found interface %s:%s", interface->ifa_name, socketToString(interface->ifa_addr, line));

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
            // Store the interface name
            strncpy(names + found * maxIfNameSize, interface->ifa_name, maxIfNameSize);
            // Store the IP address
            int salen = (family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);
            memcpy(addrs + found, interface->ifa_addr, salen);
            found++;
        }
    }

    freeifaddrs(interfaces);
    return found;
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
        SYSCHECK(setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)), "setsockopt");
#endif
    }

    // localAddr port should be 0 (Any port)
    bind(sockfd, &localAddr->sa, salen);

    /* Get the assigned Port */
    socklen_t size = salen;
    getsockname(sockfd, &localAddr->sa, &size);

#ifdef ENABLE_TRACE
    char line[1024];
    TRACE(NCCL_INIT | NCCL_NET, "Listening on socket %s", socketToString(&localAddr->sa, line));
#endif

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
    connect(*fd, &remoteAddr->sa, salen);
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

int main()
{
    return 0;
}