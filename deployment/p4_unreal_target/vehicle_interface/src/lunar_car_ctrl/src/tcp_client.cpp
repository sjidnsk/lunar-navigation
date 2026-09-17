#include <cerrno>
/**
 * @file tcp_client.cpp
 * @brief 通信层 — 车端 TCP 连接与协议包收发实现
 */

#include "lunar_car_ctrl/tcp_client.hpp"

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

namespace lunar_car_ctrl
{

LunarTcpClient::LunarTcpClient(std::string host, int port, bool big_endian,
                               const rclcpp::Logger & logger)
: host_(std::move(host))
, port_(port)
, logger_(logger)
, codec_(big_endian)
, sock_fd_(-1)
{
}

LunarTcpClient::~LunarTcpClient()
{
    disconnect();
}

bool LunarTcpClient::connect()
{
    std::lock_guard<std::mutex> lock(send_mutex_);
    sock_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd_ < 0) {
        RCLCPP_ERROR(logger_, "创建 socket 失败");
        return false;
    }

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port_);
    if (inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) <= 0) {
        RCLCPP_ERROR(logger_, "无效地址: %s", host_.c_str());
        close(sock_fd_);
        sock_fd_ = -1;
        return false;
    }

    // 设置收发超时 5s，避免阻塞
    struct timeval tv;
    tv.tv_sec = 5;
    tv.tv_usec = 0;
    setsockopt(sock_fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sock_fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    if (::connect(sock_fd_, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        RCLCPP_ERROR(logger_, "连接 %s:%d 失败", host_.c_str(), port_);
        close(sock_fd_);
        sock_fd_ = -1;
        return false;
    }

    RCLCPP_INFO(logger_, "已连接 %s:%d", host_.c_str(), port_);
    return true;
}

void LunarTcpClient::disconnect()
{
    std::lock_guard<std::mutex> lock(send_mutex_);
    if (sock_fd_ >= 0) {
        close(sock_fd_);
        sock_fd_ = -1;
        RCLCPP_INFO(logger_, "已断开连接");
    }
}

ssize_t LunarTcpClient::peekHeader(uint8_t out[4]) const
{
    if (sock_fd_ < 0) return -1;
    return recv(sock_fd_, out, 4, MSG_PEEK);
}

void LunarTcpClient::discardOneByte()
{
    uint8_t discard;
    if (sock_fd_ >= 0) {
        recv(sock_fd_, &discard, 1, 0);
    }
}

bool LunarTcpClient::recvExact(std::vector<uint8_t> & buffer, size_t size,
                               const std::atomic<bool> & running)
{
    buffer.resize(size);
    size_t received = 0;
    while (received < size && running) {
        ssize_t n = recv(sock_fd_, buffer.data() + received, size - received, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        received += n;
    }
    return running;
}

bool LunarTcpClient::recvPacket(uint32_t & command, std::vector<uint8_t> & payload,
                                const std::atomic<bool> & running)
{
    // 读取包头 16 字节: MAGIC(4) + CMD(4) + PayloadSize(4) + Checksum(4)
    std::vector<uint8_t> header_buf;
    if (!recvExact(header_buf, 16, running)) return false;

    uint32_t magic = codec_.readU32(header_buf.data());
    command = codec_.readU32(header_buf.data() + 4);
    uint32_t payload_size = codec_.readU32(header_buf.data() + 8);
    uint32_t hdr_checksum = codec_.readU32(header_buf.data() + 12);

    // 校验魔数和包头 checksum
    if (magic != MAGIC) {
        RCLCPP_WARN(logger_, "包头 MAGIC 错误: 0x%08X", magic);
        return false;
    }
    if (hdr_checksum != (magic ^ command ^ payload_size)) {
        RCLCPP_WARN(logger_, "包头校验错误");
        return false;
    }

    if (payload_size > 1024*1024) return false;

    // 读取载荷
    if (payload_size > 0) {
        if (!recvExact(payload, payload_size, running)) return false;
    } else {
        payload.clear();
    }
    return true;
}

bool LunarTcpClient::sendPacket(uint32_t command, const std::vector<uint8_t> & payload)
{
    std::lock_guard<std::mutex> lock(send_mutex_);
    if (sock_fd_ < 0) return false;

    uint32_t payload_size = static_cast<uint32_t>(payload.size());
    uint32_t checksum = MAGIC ^ command ^ payload_size;

    std::vector<uint8_t> packet;
    packet.reserve(16 + payload.size());
    codec_.pushU32(packet, MAGIC);
    codec_.pushU32(packet, command);
    codec_.pushU32(packet, payload_size);
    codec_.pushU32(packet, checksum);
    packet.insert(packet.end(), payload.begin(), payload.end());

    size_t offset = 0;
    while (offset < packet.size()) {
        const ssize_t sent = send(sock_fd_, packet.data()+offset, packet.size()-offset, MSG_NOSIGNAL);
        if (sent < 0 && errno == EINTR) continue;
        if (sent <= 0) {
            // A partial packet cannot be followed by another packet on this stream.
            shutdown(sock_fd_, SHUT_RDWR);
            RCLCPP_ERROR(logger_, "Incomplete TCP write: %zu/%zu", offset, packet.size());
            return false;
        }
        offset += static_cast<size_t>(sent);
    }
    return true;
}

} // namespace lunar_car_ctrl
