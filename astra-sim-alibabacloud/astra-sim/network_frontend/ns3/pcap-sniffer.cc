/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/* Copyright (c) 2024, cyber://A Andrea Raffaelli,
 * Le Cnam & Università di Bologna
 * PCAPNG Sniffer for NS-3 - Implementation
 */

#ifndef PCAP_SNIFFER_CC
#define PCAP_SNIFFER_CC

#include "pcap-sniffer.h"
#include "common.h" // for simulator_stop_time
#include <fstream>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cstring>
#include "ns3/simulator.h"
#include "ns3/config.h"
#include "ns3/callback.h"
#include <arpa/inet.h>

extern double simulator_stop_time; // defined in common.h

namespace ns3
{

    // -------------------- PCAP WRITER (custom) --------------------
    namespace pcap_sniffer
    {
        static bool debug_mode = false; // enable debug output
        static std::ofstream pcap_ofs;
        static std::ofstream debug_ofs;
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

        void SetOutputFile(const std::string &filename)
        {
            if (debug_mode)
            {
                // Ensure parent dir exists
                size_t pos = filename.find_last_of('/');
                if (pos != std::string::npos)
                {
                    std::string dir = filename.substr(0, pos);
                    system(("mkdir -p " + dir).c_str());
                }
                debug_ofs.open(filename, std::ios::out);
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

            pcap_ofs.open(filename, std::ios::binary | std::ios::trunc);
            if (!pcap_ofs)
                return;
            pcap_opened = true;

            // Section Header Block (SHB) – minimal mandatory fields
            std::vector<uint8_t> shb;
            auto le32 = [](uint32_t v)
            {
                return std::array<uint8_t, 4>{static_cast<uint8_t>(v),
                                              static_cast<uint8_t>(v >> 8),
                                              static_cast<uint8_t>(v >> 16),
                                              static_cast<uint8_t>(v >> 24)};
            };
            auto le16 = [](uint16_t v)
            {
                return std::array<uint8_t, 2>{static_cast<uint8_t>(v),
                                              static_cast<uint8_t>(v >> 8)};
            };
            
            shb.insert(shb.end(), le32(0x0A0D0D0A).begin(), le32(0x0A0D0D0A).end()); // Block Type
            shb.insert(shb.end(), le32(28).begin(), le32(28).end());                 // Block Total Length
            shb.insert(shb.end(), le32(0x1A2B3C4D).begin(), le32(0x1A2B3C4D).end()); // Byte‑order magic
            shb.insert(shb.end(), le32(1).begin(), le32(1).end());                   // Major version
            shb.insert(shb.end(), le32(0).begin(), le32(0).end());                   // Minor version
            shb.insert(shb.end(), le32(0xFFFFFFFFFFFFFFFFULL).begin(), le32(0xFFFFFFFFFFFFFFFFULL).end()); // Section length (-1 = unknown)
            shb.insert(shb.end(), le32(28).begin(), le32(28).end());                 // Trailing Block Total Length

            pcap_ofs.write(reinterpret_cast<const char *>(shb.data()), shb.size());

            // Interface Description Block (IDB) with proper structure
            std::vector<uint8_t> idb;
            idb.insert(idb.end(), le32(0x00000001).begin(), le32(0x00000001).end()); // Block Type
            
            // We'll fix the total length later
            size_t length_pos = idb.size();
            idb.insert(idb.end(), le32(0).begin(), le32(0).end());                   // Block Total Length (placeholder)
            
            idb.insert(idb.end(), le16(1).begin(), le16(1).end());                   // LinkType (Ethernet = 1)
            idb.insert(idb.end(), le16(0).begin(), le16(0).end());                   // Reserved
            idb.insert(idb.end(), le32(65535).begin(), le32(65535).end());           // SnapLen
            
            // Add if_tsresol option (option 9): timestamp resolution in microseconds (10^-6)
            idb.insert(idb.end(), le16(9).begin(), le16(9).end());                   // Option Code: if_tsresol
            idb.insert(idb.end(), le16(1).begin(), le16(1).end());                   // Option Length: 1 byte
            idb.push_back(6);                                                         // Value: 6 means 10^-6 (microseconds)
            idb.push_back(0); idb.push_back(0); idb.push_back(0);                   // Padding to 4-byte boundary
            
            // End of options
            idb.insert(idb.end(), le16(0).begin(), le16(0).end());                   // opt_endofopt
            idb.insert(idb.end(), le16(0).begin(), le16(0).end());                   // length = 0
            
            // Calculate and set the correct block total length
            uint32_t idb_total_length = idb.size() + 4; // +4 for trailing length field
            auto length_bytes = le32(idb_total_length);
            std::copy(length_bytes.begin(), length_bytes.end(), idb.begin() + length_pos);
            
            // Add trailing Block Total Length
            idb.insert(idb.end(), length_bytes.begin(), length_bytes.end());

            pcap_ofs.write(reinterpret_cast<const char *>(idb.data()), idb.size());
            pcap_ofs.flush();
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

            /*--- 1. Timestamp from simulation in PCAPNG format -----------------*/
            // With if_tsresol=6 (10^-6), we use microseconds directly
            uint64_t usec = Simulator::Now().GetMicroSeconds();
            uint32_t ts_high = static_cast<uint32_t>(usec >> 32);        // Upper 32 bits
            uint32_t ts_low = static_cast<uint32_t>(usec & 0xFFFFFFFF);  // Lower 32 bits

            /*--- 2. Length fields -------------------------------------*/
            uint32_t incl_len = static_cast<uint32_t>(frame.size());
            uint32_t orig_len = incl_len;

            /*--- 3. Enhanced Packet Block (EPB) -----------------------*/
            std::vector<uint8_t> epb;
            epb.reserve(32 + incl_len + ((4 - (incl_len % 4)) % 4));

            auto le32 = [](uint32_t v)
            {
                return std::array<uint8_t, 4>{
                    static_cast<uint8_t>(v),
                    static_cast<uint8_t>(v >> 8),
                    static_cast<uint8_t>(v >> 16),
                    static_cast<uint8_t>(v >> 24)};
            };

            epb.insert(epb.end(), le32(0x00000006).begin(), le32(0x00000006).end()); // Block Type (EPB)
            
            size_t length_pos = epb.size();
            epb.insert(epb.end(), le32(0).begin(), le32(0).end());                   // Placeholder for Block Total Length
            
            epb.insert(epb.end(), le32(0).begin(), le32(0).end());                   // Interface ID
            epb.insert(epb.end(), le32(ts_high).begin(), le32(ts_high).end());       // Timestamp (High)
            epb.insert(epb.end(), le32(ts_low).begin(), le32(ts_low).end());         // Timestamp (Low)
            epb.insert(epb.end(), le32(incl_len).begin(), le32(incl_len).end());     // Captured Packet Length
            epb.insert(epb.end(), le32(orig_len).begin(), le32(orig_len).end());     // Original Packet Length
            epb.insert(epb.end(), frame.begin(), frame.end());                       // Packet Data
            
            // Padding to 4-byte boundary
            while (epb.size() % 4)
                epb.push_back(0);

            // No options, so we can directly add trailing length
            // Calculate Block Total Length
            uint32_t block_len = static_cast<uint32_t>(epb.size() + 4); // +4 for trailing length
            auto len_bytes = le32(block_len);

            // Update the placeholder
            std::copy(len_bytes.begin(), len_bytes.end(), epb.begin() + length_pos);
            
            // Add trailing Block Total Length
            epb.insert(epb.end(), len_bytes.begin(), len_bytes.end());

            /*--- 4. Optional debug dump ------------------------------*/
            if (debug_mode)
            {
                debug_ofs << "PCAPNG EPB @" << Simulator::Now().GetSeconds()
                          << "s, len=" << frame.size() << " bytes, usec=" << usec << " : ";
                for (size_t i = 0; i < std::min(frame.size(), static_cast<size_t>(32)); ++i)
                    debug_ofs << std::hex << std::setw(2) << std::setfill('0')
                              << static_cast<int>(frame[i]) << ' ';
                debug_ofs << std::dec << std::endl;
            }

            /*--- 5. Write EPB to file --------------------------------*/
            pcap_ofs.write(reinterpret_cast<const char *>(epb.data()), epb.size());

            /*--- 6. Flush periodically -------------------------------*/
            static uint64_t last_flush_usec = 0;
            if (usec - last_flush_usec >= 1'000'000ULL)
            {
                pcap_ofs.flush();
                last_flush_usec = usec;
            }
        }

        // Helper: compute IPv4 header checksum (header in network byte order)
        static uint16_t ComputeIpv4HeaderChecksum(const uint8_t *ipHdr, size_t ihlBytes)
        {
            // ihlBytes must be a multiple of 2 and <= 60 (max IHL)
            uint32_t sum = 0;
            for (size_t i = 0; i < ihlBytes; i += 2)
            {
                // combine two bytes into 16-bit word (network order)
                uint16_t word = (uint16_t(ipHdr[i]) << 8) | uint16_t(ipHdr[i + 1]);
                sum += word;
            }
            // fold 32-bit sum to 16 bits
            while (sum >> 16)
                sum = (sum & 0xFFFF) + (sum >> 16);
            return static_cast<uint16_t>(~sum & 0xFFFF);
        }

        void WritePacketToPcap(Ptr<const Packet> pkt, const CustomHeader &ch)
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

            // Verifica se il buffer ha un header Ethernet valido
            auto isEthernet = [&](const std::vector<uint8_t> &buf) -> bool
            {
                if (buf.size() < 14)
                    return false;
                uint16_t ethertype = (uint16_t(buf[12]) << 8) | buf[13];
                return ethertype == 0x0800 || ethertype == 0x86DD ||
                       ethertype == 0x0806 || ethertype == 0x88B5;
            };

            // Verifica se il buffer (a un offset) contiene un header IPv4 valido
            auto isIPv4 = [&](const std::vector<uint8_t> &buf, size_t offset = 0) -> bool
            {
                if (buf.size() < offset + 20)
                    return false;
                uint8_t vihl = buf[offset];
                return (vihl >> 4) == 4 && (vihl & 0x0F) >= 5;
            };

            // Verifica se il buffer contiene un header UDP valido (porta + lunghezza)
            auto isUDP = [&](const std::vector<uint8_t> &buf, size_t offset) -> bool
            {
                if (buf.size() < offset + 8)
                    return false;
                uint16_t length = (uint16_t(buf[offset + 4]) << 8) | buf[offset + 5];
                return length >= 8 && length <= buf.size() - offset;
            };

            // Ricalcola checksum IPv4 in place
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
            bool hasUdp = false;
            size_t ipOffset = SIZE_MAX;
            size_t udpOffset = SIZE_MAX;

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

            if (hasIpv4)
            {
                uint8_t ihl = payload[ipOffset] & 0x0F;
                size_t ihlBytes = ihl * 4;
                uint8_t proto = payload[ipOffset + 9]; // protocol field
                if (proto == 17 && payload.size() >= ipOffset + ihlBytes + 8)
                {
                    udpOffset = ipOffset + ihlBytes;
                    if (isUDP(payload, udpOffset))
                        hasUdp = true;
                }
            }

            /* ---------- Fix IPv4 checksum if needed ---------- */
            if (hasIpv4)
                fixIpv4Checksum(payload, ipOffset);

            /* ---------- Build final frame ---------- */
            std::vector<uint8_t> frame;

            if (!hasEthernet)
            {
                // Synth Ethernet header
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
            if (debug_mode)
            {
                debug_ofs << "[WritePacketToPcap] "
                          << "L2=" << hasEthernet
                          << " L3=" << hasIpv4
                          << " L4(UDP)=" << hasUdp
                          << " frame.len=" << frame.size()
                          << "\n";
            }

            /* ---------- Write to PCAP ---------- */
            write_frame_with_timestamp(frame);
        }
        // Free functions used as callbacks for Config::ConnectWithoutContext
        static void _pcap_trace_cb_pkt(Ptr<const Packet> pkt, CustomHeader ch)
        {
            pcap_sniffer::WritePacketToPcap(pkt, ch);
        }

        // Helper to attach sniffer to QbbNetDevice trace sources (and some generic names)
        void AttachPcapSnifferToAllDevices(const NodeContainer &nodes, const std::string &outPath)
        {
            // Create output directory if needed
            system(("mkdir -p " + outPath.substr(0, outPath.find_last_of('/'))).c_str());
            // open pcap
            pcap_sniffer::OpenPcap(outPath);

            // Preferred: connect to QbbNetDevice explicit trace-source names (we added them).
            // wildcard path attaches only to existing trace sources.
            Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/$ns3::QbbNetDevice/PacketTx",
                                          MakeCallback(&_pcap_trace_cb_pkt));
            Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/$ns3::QbbNetDevice/PacketRx",
                                          MakeCallback(&_pcap_trace_cb_pkt));

            Simulator::Schedule(Seconds(simulator_stop_time), &pcap_sniffer::ClosePcap);
        }
    } // namespace pcap_sniffer

} // namespace ns3

#endif // PCAP_SNIFFER_CC