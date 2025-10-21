#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <string>
#include <vector>

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

} // namespace PcapUtils

#endif // TEST_UTILS_H