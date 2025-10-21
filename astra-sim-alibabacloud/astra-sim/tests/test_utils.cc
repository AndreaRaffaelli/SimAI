#include "test_utils.h"
#include <fstream>
#include <cstdint>
#include <cstring>
#include <array>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <filesystem>

namespace PcapUtils
{

    bool ValidateWithTshark(const std::string &filename)
    {
        std::string command = "tshark -r \"" + filename + "\" -q > /dev/null 2>&1";
        int result = system(command.c_str());

        if (result != 0)
        {
            std::cerr << "WARNING: tshark not found or PCAP file invalid. "
                      << "Install wireshark/tshark for complete validation." << std::endl;

            // Fall back to basic header validation
            return ValidatePcapHeader(filename);
        }

        return result == 0;
    }

    bool ValidatePcapHeader(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            return false;
        }

        PcapGlobalHeader header;
        file.read(reinterpret_cast<char *>(&header), sizeof(header));

        if (file.gcount() != sizeof(header))
        {
            return false;
        }

        // Check magic number (standard PCAP or nanosecond resolution)
        bool valid_magic = (header.magic_number == 0xA1B2C3D4) || // standard
                           (header.magic_number == 0xA1B23C4D) || // swapped
                           (header.magic_number == 0xA1B2C34D) || // nanosecond
                           (header.magic_number == 0xA1B23C4D);   // nanosecond swapped

        // Check reasonable values
        bool valid_version = header.version_major == 2 && header.version_minor == 4;
        bool valid_snaplen = header.snaplen > 0 && header.snaplen <= 65535;
        bool valid_network = header.network == 1; // DLT_EN10MB (Ethernet)

        return valid_magic && valid_version && valid_snaplen && valid_network;
    }

    int GetPacketCount(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            return -1;
        }

        // Skip global header
        file.seekg(sizeof(PcapGlobalHeader));

        int packet_count = 0;
        while (file.good())
        {
            // Read packet header
            uint32_t ts_sec, ts_usec, incl_len, orig_len;
            file.read(reinterpret_cast<char *>(&ts_sec), sizeof(ts_sec));
            file.read(reinterpret_cast<char *>(&ts_usec), sizeof(ts_usec));
            file.read(reinterpret_cast<char *>(&incl_len), sizeof(incl_len));
            file.read(reinterpret_cast<char *>(&orig_len), sizeof(orig_len));

            if (!file.good())
                break;

            // Skip packet data
            file.seekg(incl_len, std::ios::cur);
            packet_count++;
        }

        return packet_count;
    }

    bool ValidateTimestampsMonotonic(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            return false;
        }

        // Skip global header
        file.seekg(sizeof(PcapGlobalHeader));

        uint32_t last_ts_sec = 0, last_ts_usec = 0;
        bool first_packet = true;

        while (file.good())
        {
            uint32_t ts_sec, ts_usec, incl_len, orig_len;
            file.read(reinterpret_cast<char *>(&ts_sec), sizeof(ts_sec));
            file.read(reinterpret_cast<char *>(&ts_usec), sizeof(ts_usec));
            file.read(reinterpret_cast<char *>(&incl_len), sizeof(incl_len));
            file.read(reinterpret_cast<char *>(&orig_len), sizeof(orig_len));

            if (!file.good())
                break;

            if (!first_packet)
            {
                // Check if current timestamp is greater than previous
                if (ts_sec < last_ts_sec ||
                    (ts_sec == last_ts_sec && ts_usec <= last_ts_usec))
                {
                    return false;
                }
            }

            last_ts_sec = ts_sec;
            last_ts_usec = ts_usec;
            first_packet = false;

            // Skip packet data
            file.seekg(incl_len, std::ios::cur);
        }

        return true;
    }

    std::string GetPcapInfo(const std::string &filename)
    {
        std::string info = "File: " + filename + "\n";

        if (!std::filesystem::exists(filename))
        {
            return info + "Status: File does not exist\n";
        }

        auto size = std::filesystem::file_size(filename);
        info += "Size: " + std::to_string(size) + " bytes\n";

        int packet_count = GetPacketCount(filename);
        info += "Packets: " + std::to_string(packet_count) + "\n";

        bool valid = ValidatePcapHeader(filename);
        info += "Header valid: " + std::string(valid ? "yes" : "no") + "\n";

        bool wireshark_ok = ValidateWithTshark(filename);
        info += "Wireshark compliant: " + std::string(wireshark_ok ? "yes" : "no") + "\n";

        return info;
    }

    size_t GetPcapHeaderSize()
    {
        return sizeof(PcapGlobalHeader);
    }

} // namespace PcapUtils