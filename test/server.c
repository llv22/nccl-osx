#include <arpa/inet.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <netinet/in.h>

#define CLIENT_QUEUE_LEN 10
#define SERVER_PORT 7002
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

int main(void)
{
    int listen_sock_fd = -1, client_sock_fd = -1;
    socklen_t client_addr_len;
    char str_addr[INET6_ADDRSTRLEN];
    int ret, flag;
    char ch;

    /* Create socket for listening (client requests) */
    listen_sock_fd = socket(AF_INET6, SOCK_STREAM, 0);
    if (listen_sock_fd == -1)
    {
        perror("socket()");
        return EXIT_FAILURE;
    }

    /* Set socket to reuse address */
    // flag = 1;
    // int sockfd = setsockopt(listen_sock_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &flag, sizeof(flag));
    // if (sockfd == -1)
    // {
    //     perror("setsockopt()");
    //     return EXIT_FAILURE;
    // }

    server_addr.sin6_family = AF_INET6;
    // server_addr.sin6_addr = in6addr_any; //working
    char ip_str[NI_MAXHOST] = "fe80::14ab:7535:3e1e:b3c8%en0"; //working
    // char ip_str[NI_MAXHOST] = "fe80::aede:48ff:fe00:1122%en5"; //not working, no route
    // char ip_str[NI_MAXHOST] = "fe80::d08a:2aff:fee8:f776%awdl0"; //not working, no response
    // char ip_str[NI_MAXHOST] = "fe80::1ac:5dcf:2900:d44c%utun0"; //not working, no response
    inet_pton(AF_INET6, ip_str, &(server_addr.sin6_addr));                  // IP address
    server_addr.sin6_port = htons(SERVER_PORT);

    /* Bind address and socket together */
    ret = bind(listen_sock_fd, (struct sockaddr *)&server_addr, sizeof(server_addr));
    if (ret == -1)
    {
        perror("bind()");
        close(listen_sock_fd);
        return EXIT_FAILURE;
    }

    /* Get the assigned Port */
    socklen_t size = sizeof(server_addr);
    getsockname(listen_sock_fd, (struct sockaddr *)&server_addr, &size);

    /* Create listening queue (client requests) */
    ret = listen(listen_sock_fd, CLIENT_QUEUE_LEN);
    if (ret == -1)
    {
        perror("listen()");
        close(listen_sock_fd);
        return EXIT_FAILURE;
    }

    char line[1024];
    line[0] = '\0';
    socketToString((struct sockaddr *)&server_addr, line);
    line[1023] = '\0';
    printf("Listening on socket %s\n", line);

    client_addr_len = sizeof(client_addr);

    while (1)
    {
        /* Do TCP handshake with client */
        client_sock_fd = accept(listen_sock_fd,
                                (struct sockaddr *)&client_addr,
                                &client_addr_len);
        if (client_sock_fd == -1)
        {
            perror("accept()");
            close(listen_sock_fd);
            return EXIT_FAILURE;
        }

        inet_ntop(AF_INET6, &(client_addr.sin6_addr),
                  str_addr, sizeof(str_addr));
        printf("New connection from: %s:%d ...\n",
               str_addr,
               ntohs(client_addr.sin6_port));

        /* Wait for data from client */
        ret = read(client_sock_fd, &ch, 1);
        if (ret == -1)
        {
            perror("read()");
            close(client_sock_fd);
            continue;
        }

        /* Do very useful thing with received data :-) */
        ch++;

        /* Send response to client */
        ret = write(client_sock_fd, &ch, 1);
        if (ret == -1)
        {
            perror("write()");
            close(client_sock_fd);
            continue;
        }

        /* Do TCP teardown */
        ret = close(client_sock_fd);
        if (ret == -1)
        {
            perror("close()");
            client_sock_fd = -1;
        }

        printf("Connection closed\n");
    }
    return EXIT_SUCCESS;
}