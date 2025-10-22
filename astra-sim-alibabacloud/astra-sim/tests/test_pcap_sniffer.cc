#include <gtest/gtest.h>
#include <ns3/test.h>
#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/internet-module.h>
#include <ns3/csma-module.h>
#include <ns3/applications-module.h>
#include <ns3/qbb-net-device.h>
#include "system/Common.hh"
#include "test_utils.h"
#include "pcap-sniffer.h"
#include "pcap-sniffer.cc"
#include "ns3/custom-header.h"
#include <fstream>
#include <filesystem>

using namespace ns3;
using namespace ns3::pcap_sniffer;

class PcapSnifferTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create temporary directory for test files
        test_dir = "/tmp/pcap_test_" + std::to_string(time(nullptr));
        std::filesystem::create_directories(test_dir);

        // Enable debug mode for detailed output
        SetDebugMode(true);
    }

    void TearDown() override
    {
        // Clean up test files
        ClosePcap();
        if (std::filesystem::exists(test_dir))
        {
            std::filesystem::remove_all(test_dir);
        }
    }

    std::string test_dir;

    // Helper to create test nodes with QbbNetDevice
    NodeContainer CreateTestNodes(uint32_t count)
    {
        NodeContainer nodes;
        nodes.Create(count);

        // Simple CSMA network for testing
        CsmaHelper csma;
        csma.SetChannelAttribute("DataRate", StringValue("100Mbps"));
        csma.SetChannelAttribute("Delay", StringValue("1ms"));

        NetDeviceContainer devices = csma.Install(nodes);

        // Install internet stack
        InternetStackHelper internet;
        internet.Install(nodes);

        // Assign IP addresses
        Ipv4AddressHelper ipv4;
        ipv4.SetBase("10.1.1.0", "255.255.255.0");
        ipv4.Assign(devices);

        return nodes;
    }

    // Helper to create CustomHeader with UDP configuration
    CustomHeader CreateUdpHeader(uint16_t sport = 1234, uint16_t dport = 5678)
    {
        CustomHeader ch(CustomHeader::L2_Header | CustomHeader::L3_Header | CustomHeader::L4_Header);

        // Set PPP protocol (EtherType)
        ch.pppProto = 0x0800; // IPv4

        // Set IPv4 fields
        ch.m_payloadSize = 100;
        ch.ipid = 1;
        ch.m_tos = 0;
        ch.m_ttl = 64;
        ch.l3Prot = 0x11; // UDP protocol
        ch.ipv4Flags = 0;
        ch.m_fragmentOffset = 0;
        ch.sip = 0x0a010101; // 10.1.1.1
        ch.dip = 0x0a010102; // 10.1.1.2
        ch.m_headerSize = 20;

        // Set UDP fields
        ch.udp.sport = sport;
        ch.udp.dport = dport;
        ch.udp.payload_size = 100;
        ch.udp.pg = 0;
        ch.udp.seq = 1;

        return ch;
    }

    // Helper to create CustomHeader with TCP configuration
    CustomHeader CreateTcpHeader(uint16_t sport = 1234, uint16_t dport = 5678)
    {
        CustomHeader ch(CustomHeader::L2_Header | CustomHeader::L3_Header | CustomHeader::L4_Header);

        // Set PPP protocol (EtherType)
        ch.pppProto = 0x0800; // IPv4

        // Set IPv4 fields
        ch.m_payloadSize = 100;
        ch.ipid = 1;
        ch.m_tos = 0;
        ch.m_ttl = 64;
        ch.l3Prot = 0x06; // TCP protocol
        ch.ipv4Flags = 0;
        ch.m_fragmentOffset = 0;
        ch.sip = 0x0a010101; // 10.1.1.1
        ch.dip = 0x0a010102; // 10.1.1.2
        ch.m_headerSize = 20;

        // Set TCP fields
        ch.tcp.sport = sport;
        ch.tcp.dport = dport;
        ch.tcp.seq = 1;
        ch.tcp.ack = 0;
        ch.tcp.length = 5; // 20 bytes header
        ch.tcp.tcpFlags = 0;
        ch.tcp.windowSize = 65535;
        ch.tcp.urgentPointer = 0;

        return ch;
    }
};

// Test 1: Basic PCAP file creation and structure
TEST_F(PcapSnifferTest, CreatesValidPcapFile)
{
    std::string pcap_file = test_dir + "/test_basic.pcap";

    // Create and open PCAP file
    OpenPcap(pcap_file);

    // Verify file was created
    ASSERT_TRUE(std::filesystem::exists(pcap_file)) << "PCAP file was not created";

    // Verify file is not empty and has proper header
    ASSERT_TRUE(PcapUtils::ValidatePcapHeader(pcap_file))
        << "PCAP file has invalid header";

    ClosePcap();
}

// Test 2: Wireshark compliance test with UDP packet
TEST_F(PcapSnifferTest, UdpPacketIsWiresharkCompliant)
{
    std::string pcap_file = test_dir + "/test_udp_wireshark.pcap";

    OpenPcap(pcap_file);

    // Create a UDP packet and write to PCAP
    Ptr<Packet> packet = Create<Packet>(100); // 100 byte payload
    CustomHeader ch = CreateUdpHeader();

    WritePacketToPcap(packet, ch);
    ClosePcap();

    // Test with tshark (Wireshark's command-line tool)
    ASSERT_TRUE(PcapUtils::ValidateWithTshark(pcap_file))
        << "UDP PCAP file is not Wireshark compliant";
}

// Test 3: Wireshark compliance test with TCP packet
TEST_F(PcapSnifferTest, TcpPacketIsWiresharkCompliant)
{
    std::string pcap_file = test_dir + "/test_tcp_wireshark.pcap";

    OpenPcap(pcap_file);

    // Create a TCP packet and write to PCAP
    Ptr<Packet> packet = Create<Packet>(100); // 100 byte payload
    CustomHeader ch = CreateTcpHeader();

    WritePacketToPcap(packet, ch);
    ClosePcap();

    // Test with tshark
    ASSERT_TRUE(PcapUtils::ValidateWithTshark(pcap_file))
        << "TCP PCAP file is not Wireshark compliant";
}

// Test 4: Packet capture with actual network traffic
TEST_F(PcapSnifferTest, CapturesNetworkPackets)
{
    std::string pcap_file = test_dir + "/test_capture.pcap";

    NodeContainer nodes = CreateTestNodes(2);
    SetOutputFile(pcap_file);
    AttachPcapSnifferToAllDevices(nodes, pcap_file);

    // Generate some network traffic
    uint16_t port = 9; // Discard port
    PacketSinkHelper sink("ns3::UdpSocketFactory",
                          InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer sinkApp = sink.Install(nodes.Get(0));
    sinkApp.Start(Seconds(0.0));
    sinkApp.Stop(Seconds(2.0));

    OnOffHelper client("ns3::UdpSocketFactory",
                       InetSocketAddress(Ipv4Address("10.1.1.2"), port));
    client.SetAttribute("DataRate", StringValue("1Mbps"));
    client.SetAttribute("PacketSize", UintegerValue(512));

    ApplicationContainer clientApp = client.Install(nodes.Get(1));
    clientApp.Start(Seconds(1.0));
    clientApp.Stop(Seconds(1.5));

    // Run simulation
    Simulator::Stop(Seconds(3.0));
    Simulator::Run();
    Simulator::Destroy();

    // Verify capture file
    ASSERT_TRUE(std::filesystem::exists(pcap_file));
    ASSERT_GT(std::filesystem::file_size(pcap_file), 24)
        << "PCAP file should be larger than just header";

    ASSERT_TRUE(PcapUtils::ValidateWithTshark(pcap_file))
        << "Captured packets are not Wireshark compliant";

    // Verify we captured some packets
    int packet_count = PcapUtils::GetPacketCount(pcap_file);
    EXPECT_GT(packet_count, 0) << "Should have captured at least one packet";
}

// Test 5: Multiple packet types and protocols
TEST_F(PcapSnifferTest, HandlesDifferentPacketTypes)
{
    std::string pcap_file = test_dir + "/test_multiple_types.pcap";

    OpenPcap(pcap_file);

    // Test different packet sizes and protocols
    std::vector<uint32_t> packet_sizes = {64, 128, 512, 1024, 1500};

    for (uint32_t i = 0; i < packet_sizes.size(); ++i)
    {
        Ptr<Packet> packet = Create<Packet>(packet_sizes[i]);
        CustomHeader ch;

        // Alternate between UDP and TCP
        if (i % 2 == 0)
        {
            ch = CreateUdpHeader(1000 + i, 2000 + i);
        }
        else
        {
            ch = CreateTcpHeader(1000 + i, 2000 + i);
        }

        WritePacketToPcap(packet, ch);
    }

    ClosePcap();

    // Validate with tshark
    ASSERT_TRUE(PcapUtils::ValidateWithTshark(pcap_file));

    // Verify all packets were written
    int expected_count = packet_sizes.size();
    int actual_count = PcapUtils::GetPacketCount(pcap_file);
    ASSERT_EQ(actual_count, expected_count)
        << "Expected " << expected_count << " packets, got " << actual_count;
}

// Test 6: Large file handling
TEST_F(PcapSnifferTest, HandlesLargeNumberOfPackets)
{
    std::string pcap_file = test_dir + "/test_large.pcap";
    const int NUM_PACKETS = 100;

    OpenPcap(pcap_file);

    for (int i = 0; i < NUM_PACKETS; ++i)
    {
        Ptr<Packet> packet = Create<Packet>(100);
        CustomHeader ch = CreateUdpHeader(1000 + i, 2000 + i);

        WritePacketToPcap(packet, ch);
    }

    ClosePcap();

    // Verify file integrity
    ASSERT_TRUE(PcapUtils::ValidateWithTshark(pcap_file));

    int packet_count = PcapUtils::GetPacketCount(pcap_file);
    ASSERT_EQ(packet_count, NUM_PACKETS)
        << "Should have captured all " << NUM_PACKETS << " packets";
}

// Test 7: Error handling
TEST_F(PcapSnifferTest, HandlesInvalidOperations)
{
    // Try to write without opening file first
    Ptr<Packet> packet = Create<Packet>(100);
    CustomHeader ch = CreateUdpHeader();

    // This should not crash
    EXPECT_NO_THROW(WritePacketToPcap(packet, ch));

    // Try to close without opening
    EXPECT_NO_THROW(ClosePcap());
}

// Test 8: File path creation
TEST_F(PcapSnifferTest, CreatesParentDirectories)
{
    std::string pcap_file = test_dir + "/deep/nested/path/test_directories.pcap";

    // This should create all parent directories
    OpenPcap(pcap_file);

    ASSERT_TRUE(std::filesystem::exists(pcap_file))
        << "Should create parent directories automatically";

    ClosePcap();
}

// Test 9: Different CustomHeader configurations
TEST_F(PcapSnifferTest, HandlesVariousCustomHeaderConfigs)
{
    std::string pcap_file = test_dir + "/test_header_configs.pcap";

    OpenPcap(pcap_file);

    // Test 1: L2 header only
    CustomHeader ch1(CustomHeader::L2_Header);
    ch1.pppProto = 0x0800; // IPv4
    Ptr<Packet> pkt1 = Create<Packet>(50);
    WritePacketToPcap(pkt1, ch1);

    // Test 2: L2 + L3 headers
    CustomHeader ch2(CustomHeader::L2_Header | CustomHeader::L3_Header);
    ch2.pppProto = 0x0800;
    ch2.l3Prot = 0x11; // UDP
    ch2.sip = 0x0a010101;
    ch2.dip = 0x0a010102;
    Ptr<Packet> pkt2 = Create<Packet>(50);
    WritePacketToPcap(pkt2, ch2);

    // Test 3: Full L2+L3+L4 headers (UDP)
    CustomHeader ch3 = CreateUdpHeader();
    Ptr<Packet> pkt3 = Create<Packet>(50);
    WritePacketToPcap(pkt3, ch3);

    // Test 4: Full L2+L3+L4 headers (TCP)
    CustomHeader ch4 = CreateTcpHeader();
    Ptr<Packet> pkt4 = Create<Packet>(50);
    WritePacketToPcap(pkt4, ch4);

    ClosePcap();

    // Validate the mixed header file
    ASSERT_TRUE(PcapUtils::ValidateWithTshark(pcap_file));

    int packet_count = PcapUtils::GetPacketCount(pcap_file);
    ASSERT_EQ(packet_count, 4) << "Should have captured all 4 test packets";
}

// Test 10: Empty packet handling
TEST_F(PcapSnifferTest, HandlesEmptyPackets)
{
    std::string pcap_file = test_dir + "/test_empty.pcap";

    OpenPcap(pcap_file);

    // Test with zero-length packet
    Ptr<Packet> empty_packet = Create<Packet>(0);
    CustomHeader ch = CreateUdpHeader();

    WritePacketToPcap(empty_packet, ch);
    ClosePcap();

    // Should still create valid PCAP file
    ASSERT_TRUE(PcapUtils::ValidateWithTshark(pcap_file));
}

// Test 11: Verify PCAP file contains actual packet data
TEST_F(PcapSnifferTest, PcapFileContainsPacketData)
{
    std::string pcap_file = test_dir + "/test_packet_data.pcap";
    const uint32_t PACKET_SIZE = 100;

    OpenPcap(pcap_file);

    // Create packet with specific content
    uint8_t buffer[PACKET_SIZE];
    for (uint32_t i = 0; i < PACKET_SIZE; ++i)
    {
        buffer[i] = i & 0xFF; // Fill with pattern
    }
    Ptr<Packet> packet = Create<Packet>(buffer, PACKET_SIZE);
    CustomHeader ch = CreateUdpHeader();

    WritePacketToPcap(packet, ch);
    ClosePcap();

    // Verify file exists and has reasonable size
    ASSERT_TRUE(std::filesystem::exists(pcap_file));
    uintmax_t file_size = std::filesystem::file_size(pcap_file);
    EXPECT_GT(file_size, sizeof(PcapUtils::PcapGlobalHeader) + 16 + PACKET_SIZE)
        << "File should contain header + packet header + packet data";

    ASSERT_TRUE(PcapUtils::ValidateWithTshark(pcap_file));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}