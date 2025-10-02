/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/* Copyright (c) 2024, cyber://A Andrea Raffaelli,
 * Le Cnam & Università di Bologna
 * PCAP Sniffer for NS-3 - Implementation
 */

#include "pcap-sniffer.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include "ns3/simulator.h"
#include "ns3/config.h"
#include "ns3/callback.h"

extern double simulator_stop_time; // should be defined in your simulation code

namespace ns3
{

    // -------------------- PCAP WRITER (custom) --------------------
    namespace pcap_sniffer
    {

        static std::ofstream pcap_ofs;
        static uint32_t pcap_snaplen = 65535; // large snapshot length
        static bool pcap_opened = false;

        static inline void write_le_u16(std::ofstream &ofs, uint16_t v)
        {
            ofs.write(reinterpret_cast<const char *>(&v), sizeof(v));
        }

        static inline void write_le_u32(std::ofstream &ofs, uint32_t v)
        {
            ofs.write(reinterpret_cast<const char *>(&v), sizeof(v));
        }

        static inline void write_le_s32(std::ofstream &ofs, int32_t v)
        {
            ofs.write(reinterpret_cast<const char *>(&v), sizeof(v));
        }

        void OpenPcap(const std::string &filename)
        {
            if (pcap_opened)
                return;

            // Ensure parent dir exists
            size_t pos = filename.find_last_of('/');
            if (pos != std::string::npos)
            {
                std::string dir = filename.substr(0, pos);
                system(("mkdir -p " + dir).c_str());
            }

            pcap_ofs.open(filename, std::ios::binary);
            if (!pcap_ofs.is_open())
            {
                std::cerr << "PcapWriter: cannot open " << filename << std::endl;
                return;
            }

            // pcap global header (native endian). Most readers accept both orders.
            uint32_t magic_number = 0xa1b2c3d4; // file magic
            uint16_t version_major = 2;
            uint16_t version_minor = 4;
            int32_t thiszone = 0;
            uint32_t sigfigs = 0;
            uint32_t snaplen = pcap_snaplen;
            uint32_t network = 1; // LINKTYPE_ETHERNET (DLT_EN10MB)

            write_le_u32(pcap_ofs, magic_number);
            write_le_u16(pcap_ofs, version_major);
            write_le_u16(pcap_ofs, version_minor);
            write_le_s32(pcap_ofs, thiszone);
            write_le_u32(pcap_ofs, sigfigs);
            write_le_u32(pcap_ofs, snaplen);
            write_le_u32(pcap_ofs, network);

            pcap_ofs.flush();
            pcap_opened = true;
        }

        void ClosePcap()
        {
            if (pcap_opened)
            {
                pcap_ofs.flush();
                pcap_ofs.close();
                pcap_opened = false;
            }
        }

        // Helper: detect whether buffer likely already contains Ethernet header.
        // If not, prepend a small fake Ethernet header (dest: 00:00:00:00:00:01,
        // src: 00:00:00:00:00:02, ethertype 0x0800 for IPv4 or 0x86dd for IPv6).
        static bool looks_like_ethernet(const uint8_t *data, size_t len)
        {
            if (len < 14)
                return false;
            // check ethertype bytes at offset 12-13
            uint16_t ethertype = (uint16_t(data[12]) << 8) | data[13];
            if (ethertype == 0x0800 || ethertype == 0x86dd || ethertype == 0x0806)
                return true;
            return false;
        }

        static bool looks_like_ipv4_payload(const uint8_t *data, size_t len)
        {
            if (len < 1)
                return false;
            uint8_t ver = (data[0] >> 4) & 0x0f;
            return ver == 4;
        }

        static bool looks_like_ipv6_payload(const uint8_t *data, size_t len)
        {
            if (len < 1)
                return false;
            uint8_t ver = (data[0] >> 4) & 0x0f;
            return ver == 6;
        }

        void write_frame_with_timestamp(const std::vector<uint8_t> &frame)
        {
            if (!pcap_opened)
                return;

            // timestamp from ns-3 simulator
            uint64_t usec = Simulator::Now().GetMicroSeconds(); // 64-bit
            uint32_t ts_sec = static_cast<uint32_t>(usec / 1000000ull);
            uint32_t ts_usec = static_cast<uint32_t>(usec % 1000000ull);
            uint32_t incl_len = static_cast<uint32_t>(frame.size());
            uint32_t orig_len = incl_len;

            write_le_u32(pcap_ofs, ts_sec);
            write_le_u32(pcap_ofs, ts_usec);
            write_le_u32(pcap_ofs, incl_len);
            write_le_u32(pcap_ofs, orig_len);
            pcap_ofs.write(reinterpret_cast<const char *>(frame.data()), frame.size());

            // flush occasionally to avoid losing last packets on abnormal exit
            if ((Simulator::Now().GetMicroSeconds() % 1000000ull) < 1000ull)
            {
                pcap_ofs.flush();
            }
        }

        // Main writer: takes ns3::Packet and emits a pcap frame (Ethernet or faked)
        void WritePacketToPcap(Ptr<const Packet> pkt)
        {
            if (!pcap_opened)
                return;

            uint32_t size = pkt->GetSize();
            std::vector<uint8_t> payload;
            payload.resize(size);
            if (size > 0)
            {
                pkt->CopyData(payload.data(), size);
            }

            std::vector<uint8_t> frame;
            const uint8_t *data = payload.data();
            size_t len = payload.size();

            if (looks_like_ethernet(data, len))
            {
                frame = payload;
            }
            else if (looks_like_ipv4_payload(data, len))
            {
                // prepend fake ethernet header with ethertype IPv4
                frame.resize(14 + len);
                // dest MAC
                frame[0] = 0x00;
                frame[1] = 0x00;
                frame[2] = 0x00;
                frame[3] = 0x00;
                frame[4] = 0x00;
                frame[5] = 0x01;
                // src MAC
                frame[6] = 0x00;
                frame[7] = 0x00;
                frame[8] = 0x00;
                frame[9] = 0x00;
                frame[10] = 0x00;
                frame[11] = 0x02;
                // ethertype 0x0800
                frame[12] = 0x08;
                frame[13] = 0x00;
                if (len > 0)
                    memcpy(frame.data() + 14, data, len);
            }
            else if (looks_like_ipv6_payload(data, len))
            {
                frame.resize(14 + len);
                frame[0] = 0x00;
                frame[1] = 0x00;
                frame[2] = 0x00;
                frame[3] = 0x00;
                frame[4] = 0x00;
                frame[5] = 0x01;
                frame[6] = 0x00;
                frame[7] = 0x00;
                frame[8] = 0x00;
                frame[9] = 0x00;
                frame[10] = 0x00;
                frame[11] = 0x02;
                // ethertype 0x86DD
                frame[12] = 0x86;
                frame[13] = 0xDD;
                if (len > 0)
                    memcpy(frame.data() + 14, data, len);
            }
            else
            {
                // unknown: put raw payload as-is but still report as Ethernet frame (no L2)
                // create a minimal Ethernet header and put payload after it so Wireshark can still open payload bytes as "data".
                frame.resize(14 + len);
                frame[0] = 0x00;
                frame[1] = 0x00;
                frame[2] = 0x00;
                frame[3] = 0x00;
                frame[4] = 0x00;
                frame[5] = 0x01;
                frame[6] = 0x00;
                frame[7] = 0x00;
                frame[8] = 0x00;
                frame[9] = 0x00;
                frame[10] = 0x00;
                frame[11] = 0x02;
                // ethertype 0x0000 (unknown)
                frame[12] = 0x00;
                frame[13] = 0x00;
                if (len > 0)
                    memcpy(frame.data() + 14, data, len);
            }

            write_frame_with_timestamp(frame);
        }

    } // namespace pcap_sniffer

    // Free functions used as callbacks for Config::ConnectWithoutContext
    static void _pcap_trace_cb_pkt(Ptr<const Packet> pkt)
    {
        pcap_sniffer::WritePacketToPcap(pkt);
    }

    static void _pcap_trace_cb_pkt_with_dev(Ptr<const Packet> pkt, Ptr<NetDevice> dev)
    {
        pcap_sniffer::WritePacketToPcap(pkt);
    }

    static void _pcap_trace_cb_pkt_with_addr(Ptr<const Packet> pkt, const Address &addr)
    {
        pcap_sniffer::WritePacketToPcap(pkt);
    }

    static void _pcap_trace_cb_pkt_with_node(Ptr<const Packet> pkt, Ptr<Node> node, uint32_t ifindex)
    {
        pcap_sniffer::WritePacketToPcap(pkt);
    }

    // Helper to attach sniffer to QbbNetDevice trace sources (and some generic names)
    void AttachPcapSnifferToAllDevices(const NodeContainer &nodes, const std::string &outPath)
    {
        // open pcap
        pcap_sniffer::OpenPcap(outPath);

        // Preferred: connect to QbbNetDevice explicit trace-source names (we added them).
        // wildcard path attaches only to existing trace sources.
        Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/$ns3::QbbNetDevice/PacketTx",
                                      MakeCallback(&_pcap_trace_cb_pkt));
        Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/$ns3::QbbNetDevice/PacketRx",
                                      MakeCallback(&_pcap_trace_cb_pkt));

        // also attempt to connect to common trace-source names so we capture other device types
        // these calls will attach only where that trace source exists
        Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/MacTx",
                                      MakeCallback(&_pcap_trace_cb_pkt));
        Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/MacRx",
                                      MakeCallback(&_pcap_trace_cb_pkt));
        Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/PhyTxBegin",
                                      MakeCallback(&_pcap_trace_cb_pkt));
        Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/PhyRxEnd",
                                      MakeCallback(&_pcap_trace_cb_pkt));
        Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/Tx",
                                      MakeCallback(&_pcap_trace_cb_pkt));
        Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/Rx",
                                      MakeCallback(&_pcap_trace_cb_pkt));
        
        extern double simulator_stop_time;
        Simulator::Schedule(Seconds(simulator_stop_time), &pcap_sniffer::ClosePcap);
    }

} // namespace ns3