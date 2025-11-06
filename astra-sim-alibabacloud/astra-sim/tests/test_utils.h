#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <string>
#include <vector>
#include <cstdint>  
#include <ns3/test.h>
#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/internet-module.h>
#include <ns3/csma-module.h>
#include <ns3/applications-module.h>
#include <ns3/qbb-net-device.h>
#include "ns3/custom-header.h"
#include "ns3/ptr.h"
#include "ns3/packet.h"
#include "ns3/node-container.h"
#include "ns3/net-device.h"
#include "ns3/address.h"
#include "ns3/node.h"
#include "ns3/ipv4.h"           // ADD THIS
#include "ns3/ipv4-address.h"   // ADD THIS

namespace PcapUtils
{
    struct PcapGlobalHeader
    {
        uint32_t magic_number;  // magic number
        uint16_t version_major; // major version number
        uint16_t version_minor; // minor version number
        int32_t thiszone;       // GMT to local correction
        uint32_t sigfigs;       // accuracy of timestamps
        uint32_t snaplen;       // max length of captured packets, in octets
        uint32_t network;       // data link type
    };

    /**
     * \brief Validate PCAP file header using tshark
     * \param filename Path to PCAP file
     * \return true if file is valid and Wireshark compliant
     */
    bool ValidateWithTshark(const std::string &filename);

    /**
     * \brief Validate basic PCAP file header structure
     * \param filename Path to PCAP file
     * \return true if header is valid
     */
    bool ValidatePcapHeader(const std::string &filename);

    /**
     * \brief Get number of packets in PCAP file using tshark
     * \param filename Path to PCAP file
     * \return Number of packets, -1 on error
     */
    int GetPacketCount(const std::string &filename);

    /**
     * \brief Validate that packet timestamps are monotonic
     * \param filename Path to PCAP file
     * \return true if timestamps are strictly increasing
     */
    bool ValidateTimestampsMonotonic(const std::string &filename);

    /**
     * \brief Get detailed information about PCAP file
     * \param filename Path to PCAP file
     * \return String with file information
     */
    std::string GetPcapInfo(const std::string &filename);

    /**
     * @brief Create a network topology with QbbNetDevice connections for testing
     *
     * @param numNodes Number of nodes to create
     * @param dataRate Data rate for the links (e.g., "10Gbps", "25Gbps", "100Gbps")
     * @param linkDelay Link delay (e.g., "10us", "100ns")
     * @param errorRate Packet error rate (0.0 for no errors)
     * @return NodeContainer with configured nodes and QbbNetDevice connections
     */
    ns3::NodeContainer CreateQbbTestTopology(
        uint32_t numNodes,
        const std::string &dataRate = "10Gbps",
        const std::string &linkDelay = "10us",
        double errorRate = 0.0);

    /**
     * @brief Create a simple two-node QbbNetDevice topology for basic tests
     *
     * @param dataRate Data rate for the link (default: "10Gbps")
     * @param linkDelay Link delay (default: "10us")
     * @return NodeContainer with 2 nodes connected by QbbNetDevice
     */
    ns3::NodeContainer CreateSimpleQbbTopology(
        const std::string &dataRate = "10Gbps",
        const std::string &linkDelay = "10us");

    /**
     * @brief Get the IP address of a node
     *
     * @param node The node to get the address from
     * @param interface Interface index (default: 1, as 0 is loopback)
     * @return Ipv4Address The IP address of the node
     */
    ns3::Ipv4Address GetNodeAddress(ns3::Ptr<ns3::Node> node, uint32_t interface = 1);
} // namespace PcapUtils

#endif // TEST_UTILS_H