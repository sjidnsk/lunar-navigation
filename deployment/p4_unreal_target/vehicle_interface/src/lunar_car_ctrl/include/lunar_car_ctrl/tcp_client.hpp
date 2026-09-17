/**
 * @file tcp_client.hpp
 * @brief 通信层 — 车端 TCP 连接与协议包收发
 *
 * 职责:
 *   - TCP 连接建立/断开 (超时 5s)
 *   - 精确接收 / 协议包接收 (包头 16B + payload)
 *   - peek 首 4 字节供上层判断数据包格式
 *   - 协议包发送 (线程安全)
 *
 * 不负责: 重连策略、数据解析、业务调度 (由调度层处理)
 */

#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <cstdint>
#include <sys/types.h>

#include <rclcpp/rclcpp.hpp>
#include "lunar_car_ctrl/protocol_codec.hpp"

namespace lunar_car_ctrl
{

/**
 * @brief 车端 TCP 客户端
 */
class LunarTcpClient
{
public:
    LunarTcpClient(std::string host, int port, bool big_endian,
                   const rclcpp::Logger & logger);
    ~LunarTcpClient();

    // 禁止拷贝 (持有 socket 资源)
    LunarTcpClient(const LunarTcpClient &) = delete;
    LunarTcpClient & operator=(const LunarTcpClient &) = delete;

    /** @brief 建立 TCP 连接 (收发超时 5s) */
    bool connect();

    /** @brief 断开 TCP 连接 */
    void disconnect();

    /** @brief 连接状态查询 */
    bool isConnected() const { return sock_fd_ >= 0; }

    /** @brief peek 前 4 字节 (不消费数据), 返回实际读取字节数 */
    ssize_t peekHeader(uint8_t out[4]) const;

    /** @brief 丢弃 1 字节 (流重新同步用) */
    void discardOneByte();

    /** @brief 循环 recv 直到精确接收 size 字节, running=false 时中止 */
    bool recvExact(std::vector<uint8_t> & buffer, size_t size,
                   const std::atomic<bool> & running);

    /** @brief 接收一个完整协议包: 包头(16B) + 载荷 */
    bool recvPacket(uint32_t & command, std::vector<uint8_t> & payload,
                    const std::atomic<bool> & running);

    /** @brief 构建并发送协议包 (线程安全) */
    bool sendPacket(uint32_t command, const std::vector<uint8_t> & payload);

    /** @brief 获取编解码器 (供上层解析数据使用) */
    const ProtocolCodec & codec() const { return codec_; }

private:
    std::string host_;           ///< 车端服务器 IP
    int port_;                   ///< 车端服务器端口
    rclcpp::Logger logger_;      ///< 日志记录器 (来自调度层节点)
    ProtocolCodec codec_;        ///< 字节序编解码器
    int sock_fd_;                ///< TCP socket 文件描述符
    std::mutex send_mutex_;      ///< 发送互斥锁 (防止并发发送)
};

} // namespace lunar_car_ctrl
