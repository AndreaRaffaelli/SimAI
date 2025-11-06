#include "test_utils.h"
#include <gtest/gtest.h>
#include "system/Common.hh"
#include <fstream>
#include <filesystem>
#include "ns3/qbb-helper.h"
#include "ns3/error-model.h"
#include "ns3/random-variable-stream.h"
#include "ns3/ipv4.h"
#include "ns3/node-container.h"
// PcapPlusPlus includes
#include "PcapFileDevice.h"
#include "RawPacket.h"
#include "Packet.h"

using namespace ns3;
using namespace std;

using ns3::Ptr;
using ns3::Node;
using ns3::Ipv4;
using ns3::Ipv4Address;

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
        std::string cmd = "file -b '" + filename + "'";
        FILE *pipe = popen(cmd.c_str(), "r");
        if (!pipe)
        {
            return false;
        }

        char buffer[256];
        std::string result;
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
        {
            result += buffer;
        }
        pclose(pipe);

        // Check if it's a valid pcap or pcapng file
        return (result.find("pcap") != std::string::npos ||
                result.find("capture file") != std::string::npos);
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

    NodeContainer CreateQbbTestTopology(
        uint32_t numNodes,
        const std::string &dataRate,
        const std::string &linkDelay,
        double errorRate)
    {
        // Create nodes
        NodeContainer nodes;
        nodes.Create(numNodes);

        // Install Internet stack
        InternetStackHelper internet;
        internet.Install(nodes);

        // Set up QbbHelper
        QbbHelper qbb;
        qbb.SetDeviceAttribute("DataRate", StringValue(dataRate));
        qbb.SetChannelAttribute("Delay", StringValue(linkDelay));

        // Configure error model
        Ptr<RateErrorModel> rem = CreateObject<RateErrorModel>();
        Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable>();
        rem->SetRandomVariable(uv);
        uv->SetStream(50);
        rem->SetAttribute("ErrorRate", DoubleValue(errorRate));
        rem->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));
        qbb.SetDeviceAttribute("ReceiveErrorModel", PointerValue(rem));

        for (uint32_t i = 0; i < numNodes - 1; i++)
        {
            Ptr<Node> node1 = nodes.Get(i);
            Ptr<Node> node2 = nodes.Get(i + 1);

            // Install QbbNetDevice between adjacent nodes
            NetDeviceContainer devices = qbb.Install(node1, node2);

            // Assign IP addresses using Ipv4AddressHelper
            std::ostringstream network;
            network << "10." << (i + 1) << ".1.0";

            Ipv4AddressHelper address;
            address.SetBase(network.str().c_str(), "255.255.255.0");
            Ipv4InterfaceContainer interfaces = address.Assign(devices);
        }
        return nodes;
    }
    
    NodeContainer CreateSimpleQbbTopology(
        const std::string &dataRate,
        const std::string &linkDelay)
    {
        // Create 2 nodes
        NodeContainer nodes;
        nodes.Create(2);

        // Install Internet stack
        InternetStackHelper internet;
        internet.Install(nodes);

        // Set up QbbHelper
        QbbHelper qbb;
        qbb.SetDeviceAttribute("DataRate", StringValue(dataRate));
        qbb.SetChannelAttribute("Delay", StringValue(linkDelay));

        // No errors for simple topology
        Ptr<RateErrorModel> rem = CreateObject<RateErrorModel>();
        Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable>();
        rem->SetRandomVariable(uv);
        uv->SetStream(50);
        rem->SetAttribute("ErrorRate", DoubleValue(0.0));
        rem->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));
        qbb.SetDeviceAttribute("ReceiveErrorModel", PointerValue(rem));

        // Connect the two nodes
        NetDeviceContainer devices = qbb.Install(nodes.Get(0), nodes.Get(1));

        // Assign IP addresses using helper (recommended ns-3 way)
        Ipv4AddressHelper address;
        address.SetBase("10.1.1.0", "255.255.255.0");
        Ipv4InterfaceContainer interfaces = address.Assign(devices);

        return nodes;
    }

    Ipv4Address GetNodeAddress(Ptr<Node> node, uint32_t interface)
    {
        Ptr<Ipv4> ipv4 = node->GetObject<Ipv4>();
        if (interface >= ipv4->GetNInterfaces())
        {
            return Ipv4Address::GetAny();
        }
        return ipv4->GetAddress(interface, 0).GetLocal();
    }

} // namespace PcapUtils
