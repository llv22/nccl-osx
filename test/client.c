#include <arpa/inet.h>
#include <netdb.h>
#include <string.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>

#define SERVER_PORT 7002
#define CLIENT_QUEUE_LEN 10
#define	NI_MAXHOST	1025
#define	NI_MAXSERV	32

struct sockaddr_in6 server_addr, client_addr;

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
  (void)getnameinfo(saddr, sizeof(server_addr), host, NI_MAXHOST, service, NI_MAXSERV, NI_NUMERICHOST | NI_NUMERICSERV);
  sprintf(buf, "%s<%s>", host, service);
  return buf;
}

//see: https://gist.githubusercontent.com/jirihnidek/388271b57003c043d322/raw/010c29eb416508c217605666994ddb56a63a0c80/client.c
int main(int argc, char *argv[])
{
    int sock_fd = -1;
    int ret;
    char ch = 'a';

    /* Arguments could be used in getaddrinfo() to get e.g. IP of server */
    (void)argc;
    (void)argv;

    /* Create socket for communication with server */
    sock_fd = socket(AF_INET6, SOCK_STREAM, 0);
    if (sock_fd == -1)
    {
        perror("socket()");
        return EXIT_FAILURE;
    }

    /* Connect to server running on localhost */
    server_addr.sin6_family = AF_INET6;
    // ip: working
    inet_pton(AF_INET6, "fe80::14ab:7535:3e1e:b3c8%en0", &server_addr.sin6_addr);
    // ip: not working, reason: No route to host
    // inet_pton(AF_INET6, "fe80::aede:48ff:fe00:1122%en5", &server_addr.sin6_addr);
    // ip: not working, reason: no response
    // inet_pton(AF_INET6, "fe80::d08a:2aff:fee8:f776%awdl0", &server_addr.sin6_addr);
    // ip: not working, reason: no response
    // inet_pton(AF_INET6, "fe80::1ac:5dcf:2900:d44c%utun0", &server_addr.sin6_addr);
    // inet_pton(AF_INET6, "::1", &server_addr.sin6_addr);
    server_addr.sin6_port = htons(SERVER_PORT);

    char line[1024];
    line[0] = '\0';
    socketToString((struct sockaddr *)&server_addr, line);
    line[1023] = '\0';
    printf("Connecting to socket %s\n", line);

    const int one = 1;
    setsockopt(sock_fd, IPPROTO_TCP, TCP_NODELAY, (char *)&one, sizeof(int));

    /* Try to do TCP handshake with server */
    ret = connect(sock_fd, (struct sockaddr *)&server_addr, sizeof(server_addr));
    if (ret == -1)
    {
        perror("connect()");
        fprintf(stderr, "errno - %s\n", strerror(errno));
        close(sock_fd);
        return EXIT_FAILURE;
    }

    /* Send data to server */
    ret = write(sock_fd, &ch, 1);
    if (ret == -1)
    {
        perror("write");
        close(sock_fd);
        return EXIT_FAILURE;
    }

    /* Wait for data from server */
    ret = read(sock_fd, &ch, 1);
    if (ret == -1)
    {
        perror("read()");
        close(sock_fd);
        return EXIT_FAILURE;
    }

    printf("Received %c from server\n", ch);

    /* DO TCP teardown */
    ret = close(sock_fd);
    if (ret == -1)
    {
        perror("close()");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}