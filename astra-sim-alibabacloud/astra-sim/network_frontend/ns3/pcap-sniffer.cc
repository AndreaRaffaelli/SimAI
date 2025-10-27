/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

/* Copyright (c) 2024, cyber://A Andrea Raffaelli,
 * Le Cnam & Università di Bologna
 * PCAPNG Sniffer for NS-3 - Implementation (Using PcapPlusPlus)
 */

#ifndef PCAP_SNIFFER_CC
#define PCAP_SNIFFER_CC
#include "pcap-sniffer.h"

#include "common.h" // for simulator_stop_time

#include <iostream>
#include <fstream>
#include <memory>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <arpa/inet.h> // for htons, htonl
#include "ns3/simulator.h"
#include "ns3/config.h"
#include "ns3/callback.h"

// PcapPlusPlus includes
#include "PcapFileDevice.h"
#include "RawPacket.h"
#include "Packet.h"

extern double simulator_stop_time; // defined in common.h

namespace ns3
{

    namespace pcap_sniffer
    {

        // Static variables
        static bool debug_mode = false;
        static std::ofstream debug_ofs;
        static std::unique_ptr<pcpp::PcapNgFileWriterDevice> pcap_writer;
        static bool pcap_opened = false;
        static std::string output_filename;

        void SetOutputFile(const std::string &filename)
        {
            output_filename = filename;

            if (debug_mode)
            {
                // Ensure parent dir exists
                size_t pos = filename.find_last_of('/');
                if (pos != std::string::npos)
                {
                    std::string dir = filename.substr(0, pos);
                    system(("mkdir -p " + dir).c_str());
                }

                debug_ofs.open(filename + ".debug", std::ios::out);
                if (!debug_ofs.is_open())
                {
                    std::cerr << "PcapSniffer: cannot open debug file " << filename << std::endl;
                }
            }
        }

        void SetDebugMode(bool enable)
        {
            debug_mode = enable;
            if (debug_mode && !debug_ofs.is_open())
            {
                debug_ofs.open("pcap_sniffer.debug", std::ios::out);
                if (!debug_ofs.is_open())
                {
                    std::cerr << "PcapSniffer: cannot open default debug file pcap_sniffer.debug" << std::endl;
                    debug_mode = false;
                }
            }
        }

        void OpenPcap(const std::string &filename)
        {
            // Ensure parent directory exists
            size_t pos = filename.find_last_of('/');
            if (pos != std::string::npos)
            {
                std::string dir = filename.substr(0, pos);
                system(("mkdir -p " + dir).c_str());
            }

            // Create PcapNg file writer using PcapPlusPlus
            pcap_writer = std::make_unique<pcpp::PcapNgFileWriterDevice>(filename.c_str());

            if (!pcap_writer->open())
            {
                std::cerr << "PcapSniffer: cannot open file " << filename << " for writing" << std::endl;
                pcap_writer.reset();
                return;
            }

            // CRITICAL: Flush to write the Section Header Block to disk
            pcap_writer->flush();

            pcap_opened = true;

            if (debug_mode && debug_ofs.is_open())
            {
                debug_ofs << "PcapSniffer: opened file " << filename << " successfully" << std::endl;
            }
        }

        void ClosePcap()
        {
            if (pcap_opened && pcap_writer)
            {
                pcap_writer->close();
                pcap_writer.reset();
                pcap_opened = false;

                if (debug_mode && debug_ofs.is_open())
                {
                    debug_ofs << "PcapSniffer: closed PCAP file" << std::endl;
                }
            }

            if (debug_ofs.is_open())
            {
                debug_ofs.close();
            }
        }

        // Helper: detect whether buffer likely already contains Ethernet header
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

        // Helper: compute IPv4 header checksum
        static uint16_t ComputeIpv4HeaderChecksum(const uint8_t *ipHdr, size_t ihlBytes)
        {
            uint32_t sum = 0;
            for (size_t i = 0; i < ihlBytes; i += 2)
            {
                uint16_t word = (uint16_t(ipHdr[i]) << 8) | uint16_t(ipHdr[i + 1]);
                sum += word;
            }

            while (sum >> 16)
                sum = (sum & 0xFFFF) + (sum >> 16);

            return static_cast<uint16_t>(~sum & 0xFFFF);
        }

        void write_frame_with_timestamp(const std::vector<uint8_t> &frame)
        {
            if (!pcap_opened || !pcap_writer)
                return;

            // Get timestamp from NS-3 simulator
            uint64_t usec = Simulator::Now().GetMicroSeconds();

            // Convert to timespec (PcapPlusPlus uses timespec for timestamps)
            timespec ts;
            ts.tv_sec = usec / 1000000;
            ts.tv_nsec = (usec % 1000000) * 1000; // Convert microseconds to nanoseconds

            // Create RawPacket with Ethernet link type
            pcpp::RawPacket rawPacket(
                frame.data(),
                frame.size(),
                ts,
                false,
                pcpp::LINKTYPE_ETHERNET);

            // Write packet using PcapPlusPlus
            if (!pcap_writer->writePacket(rawPacket))
            {
                if (debug_mode && debug_ofs.is_open())
                {
                    debug_ofs << "PcapSniffer: failed to write packet at "
                              << Simulator::Now().GetSeconds() << "s" << std::endl;
                }
                return;
            }

            // Optional debug output
            if (debug_mode && debug_ofs.is_open())
            {
                debug_ofs << "PCAPNG Packet @" << Simulator::Now().GetSeconds()
                          << "s, len=" << frame.size() << " bytes, usec=" << usec << std::endl;
            }

            // Flush periodically (every second of simulation time)
            static uint64_t last_flush_usec = 0;
            if (usec - last_flush_usec >= 1000000ULL)
            {
                pcap_writer->flush();
                last_flush_usec = usec;
            }
        }

        void WritePacketToPcap(ns3::Ptr<ns3::Packet const> pkt, ns3::CustomHeader const &ch)
        {
            NS_ASSERT_MSG(pkt != nullptr, "Null packet passed to WritePacketToPcap");

            if (!pcap_opened)
                return;

            /* ---------- Extract payload ---------- */
            size_t hdSize = ch.GetSerializedSize();
            std::vector<uint8_t> hdBuf(hdSize);
            if (hdSize > 0)
            {
                Buffer buffer;
                buffer.AddAtEnd(hdSize);
                Buffer::Iterator it = buffer.Begin();
                ch.Serialize(it);
                buffer.CopyData(hdBuf.data(), hdSize);
            }

            uint32_t plSize = pkt->GetSize();
            std::vector<uint8_t> payload(plSize);
            if (plSize)
                pkt->CopyData(payload.data(), plSize);

            /* ---------- Helper lambdas ---------- */
            auto isEthernet = [&](const std::vector<uint8_t> &buf) -> bool
            {
                if (buf.size() < 14)
                    return false;

                uint16_t ethertype = (uint16_t(buf[12]) << 8) | buf[13];
                return ethertype == 0x0800 || ethertype == 0x86DD ||
                       ethertype == 0x0806 || ethertype == 0x88B5;
            };

            auto isIPv4 = [&](const std::vector<uint8_t> &buf, size_t offset = 0) -> bool
            {
                if (buf.size() < offset + 20)
                    return false;

                uint8_t vihl = buf[offset];
                return (vihl >> 4) == 4 && (vihl & 0x0F) >= 5;
            };

            auto fixIpv4Checksum = [&](std::vector<uint8_t> &buf, size_t offset)
            {
                uint8_t ihl = buf[offset] & 0x0F;
                size_t ihlBytes = ihl * 4;
                if (ihlBytes >= 20 && offset + ihlBytes <= buf.size())
                {
                    buf[offset + 10] = 0;
                    buf[offset + 11] = 0;
                    uint16_t csum = ComputeIpv4HeaderChecksum(buf.data() + offset, ihlBytes);
                    uint16_t csumNet = htons(csum);
                    buf[offset + 10] = static_cast<uint8_t>(csumNet >> 8);
                    buf[offset + 11] = static_cast<uint8_t>(csumNet & 0xFF);
                }
            };

            /* ---------- Layer detection ---------- */
            bool hasEthernet = isEthernet(payload);
            if (isEthernet(hdBuf))
            {
                hasEthernet = true;
                // copy hdBuf at the beginning of payload
                if (payload.size() >= hdBuf.size())
                {
                    std::memcpy(payload.data(), hdBuf.data(), hdBuf.size());
                }
            }

            bool hasIpv4 = false;
            size_t ipOffset = SIZE_MAX;

            if (hasEthernet && payload.size() >= 14)
            {
                ipOffset = 14;
                if (isIPv4(payload, ipOffset))
                    hasIpv4 = true;
            }
            else if (isIPv4(payload, 0))
            {
                ipOffset = 0;
                hasIpv4 = true;
            }

            /* ---------- Fix IPv4 checksum if needed ---------- */
            if (hasIpv4)
                fixIpv4Checksum(payload, ipOffset);

            /* ---------- Build final frame ---------- */
            std::vector<uint8_t> frame;
            if (!hasEthernet)
            {
                // Synthesize Ethernet header
                uint16_t ethertype = hasIpv4 ? 0x0800 : 0x88B5;
                frame.assign(14 + payload.size(), 0);

                // dest MAC = 00:00:00:00:00:01
                frame[5] = 1;
                // src MAC = 00:00:00:00:00:02
                frame[11] = 2;
                // ethertype
                frame[12] = static_cast<uint8_t>(ethertype >> 8);
                frame[13] = static_cast<uint8_t>(ethertype & 0xFF);

                if (!payload.empty())
                    std::memcpy(frame.data() + 14, payload.data(), payload.size());
            }
            else
            {
                frame = std::move(payload);
            }

            /* ---------- Optional diagnostics ---------- */
            if (debug_mode && debug_ofs.is_open())
            {
                debug_ofs << "[WritePacketToPcap] "
                          << "L2=" << hasEthernet
                          << " L3=" << hasIpv4
                          << " frame.len=" << frame.size()
                          << std::endl;
            }

            /* ---------- Write to PCAP ---------- */
            write_frame_with_timestamp(frame);
        }

        // Free functions used as callbacks for Config::ConnectWithoutContext
        static void _pcap_trace_cb_pkt(Ptr<const Packet> pkt, CustomHeader ch)
        {
            pcap_sniffer::WritePacketToPcap(pkt, ch);
        }

        void AttachPcapSnifferToAllDevices(const NodeContainer &nodes, const std::string &outPath)
        {
            // Create output directory if needed
            size_t pos = outPath.find_last_of('/');
            if (pos != std::string::npos)
            {
                std::string dir = outPath.substr(0, pos);
                system(("mkdir -p " + dir).c_str());
            }

            // Open pcap
            pcap_sniffer::OpenPcap(outPath);

            // Iterate through all nodes in the container
            for (uint32_t i = 0; i < nodes.GetN(); ++i)
            {
                Ptr<Node> node = nodes.Get(i);

                // Iterate through all devices on this node
                for (uint32_t j = 0; j < node->GetNDevices(); ++j)
                {
                    Ptr<NetDevice> device = node->GetDevice(j);
                    Ptr<QbbNetDevice> qbbDevice = device->GetObject<QbbNetDevice>();
                    if (qbbDevice)
                    {
                        // Connect to PacketTx trace source
                        qbbDevice->TraceConnectWithoutContext(
                            "PacketTx",
                            MakeCallback(&_pcap_trace_cb_pkt));

                        // Connect to PacketRx trace source
                        qbbDevice->TraceConnectWithoutContext(
                            "PacketRx",
                            MakeCallback(&_pcap_trace_cb_pkt));

                        if (debug_mode && debug_ofs.is_open())
                        {
                            debug_ofs << "PcapSniffer: Attached to Node " << node->GetId()
                                      << " Device " << j << " (QbbNetDevice)" << std::endl;
                        }
                    }
                    else
                    {
                        // If not a QbbNetDevice, try generic NetDevice trace sources
                        // You can add support for other device types here
                        if (debug_mode && debug_ofs.is_open())
                        {
                            debug_ofs << "PcapSniffer: Skipped Node " << node->GetId()
                                      << " Device " << j << " (not a QbbNetDevice)" << std::endl;
                        }
                    }
                }
            }

            // Schedule PCAP file closure
            Simulator::Schedule(Seconds(simulator_stop_time), &pcap_sniffer::ClosePcap);

            if (debug_mode && debug_ofs.is_open())
            {
                debug_ofs << "PcapSniffer: Attached to " << nodes.GetN()
                          << " nodes, PCAP will close at " << simulator_stop_time << "s" << std::endl;
            }
        }

    } // namespace pcap_sniffer

} // namespace ns3
#endif // PCAP_SNIFFER_CC
