/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/* Copyright (c) 2024, cyber://A Andrea Raffaelli,
 * Le Cnam & Università di Bologna
 *
 * PCAP Sniffer for NS-3
 *
 * This utility provides PCAP packet capture functionality for NS-3 simulations.
 * It captures packets from QbbNetDevice and other network devices and writes
 * them to a PCAP file that can be analyzed with tools like Wireshark.
 */

#ifndef PCAP_SNIFFER_H
#define PCAP_SNIFFER_H

#include <string>
#include <vector>
#include <cstdint>
#include "ns3/ptr.h"
#include "ns3/packet.h"
#include "ns3/node-container.h"
#include "ns3/net-device.h"
#include "ns3/address.h"
#include "ns3/node.h"

namespace ns3
{

    namespace pcap_sniffer
    {

        /**
         * \brief Open a PCAP file for writing
         * \param filename The path to the output PCAP file
         *
         * Creates necessary parent directories and writes PCAP global header.
         * Uses Ethernet (DLT_EN10MB) link type.
         */
        void OpenPcap(const std::string &filename);

        /**
         * \brief Close the currently open PCAP file
         *
         * Flushes and closes the output file stream.
         */
        void ClosePcap();

        /**
         * \brief Write a packet frame with timestamp to PCAP
         * \param frame The Ethernet frame data to write
         *
         * Internal function that writes a complete PCAP packet record
         * with timestamp from NS-3 simulator.
         */
        void write_frame_with_timestamp(const std::vector<uint8_t> &frame);

        /**
         * \brief Write an NS-3 packet to the PCAP file
         * \param pkt Pointer to the packet to capture
         *
         * Extracts packet data and writes it to PCAP with proper Ethernet framing.
         * Automatically detects if Ethernet header is present or needs to be added.
         * Supports IPv4, IPv6, and raw payloads.
         */
        void WritePacketToPcap(Ptr<const Packet> pkt, CustomHeader ch);

        /**
         * \brief Set out file path for PCAP output
         * \param filename The path to the output PCAP file
         *
         * This function:
         * - Sets the output file path for PCAP captures
         */
        void SetOutputFile(const std::string &filename);

        /**
         * \brief Set debug mode for PCAP sniffer
         * \param enable true to enable debug output, false to disable
         *
         * This function:
         * - Enables or disables debug output for the PCAP sniffer
         */
        void SetDebugMode(bool enable);

        /**
         * \brief Attach PCAP sniffer to all devices in a node container
         * \param nodes The nodes whose devices should be captured
         * \param outPath Path to the output PCAP file
         *
         * This function:
         * - Opens the PCAP file
         * - Connects to PacketTx/PacketRx trace sources on QbbNetDevice
         * - Connects to common trace sources on other device types
         * - Schedules PCAP file closure after simulation end
         */
        void AttachPcapSnifferToAllDevices(const NodeContainer &nodes, const std::string &outPath);

    } // namespace pcap_sniffer

} // namespace ns3

#endif // PCAP_SNIFFER_H
