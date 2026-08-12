#include "functions.h"

// Decoded fields of a single order-book packet (see decodePacket()).
struct OrderBookEntry {
    uint32_t stockID;
    bool orderType;      // true = sell, false = buy
    uint8_t orderQty;
    uint8_t orderValue;
};

// Unpacks a de-stuffed 49-bit packet into its four fields.
// Layout (LSB first): stockID[0:32) | orderType[32] | orderQty[33:41) | orderValue[41:49)
OrderBookEntry decodePacket(uint64_t packet) {
    OrderBookEntry entry;
    entry.stockID = (packet >> 0) & 0xFFFFFFFF;
    entry.orderType = ((packet >> 32) & 0x1) == 1;
    entry.orderQty = (packet >> 33) & 0xFF;
    entry.orderValue = (packet >> 41) & 0xFF;

    return entry;
}

// Bit-destuffing: the raw feed inserts one filler bit after every run of 5
// consecutive 1-bits (a standard framing technique to bound run lengths for
// clock recovery). This walks the raw 64-bit packet, copies bits through to
// `result` normally, and whenever the 5th consecutive 1-bit has just been
// copied, skips the next bit (the inserted filler) before continuing. Stops
// once 49 real payload bits have been recovered.
uint64_t unstuffBits(uint64_t packet) {
    uint64_t result = 0;
    int outputBitPos = 0;
    int consecutiveOnes = 0;
    const int TARGET_BITS = 49;

    int i = 0;
    while (i < 64 && outputBitPos < TARGET_BITS) {
        bool bit = (packet >> i) & 0x1;

        if (bit) {
            consecutiveOnes++;
            result |= (1ULL << outputBitPos);
            outputBitPos++;

            if (consecutiveOnes == 5) {
                consecutiveOnes = 0;
                i+=2;  // skip the stuffed filler bit that follows a run of 5
                continue;
            }
        } else {
            consecutiveOnes = 0;
            outputBitPos++;
        }
        i++;
    }

    return result;
}

// Running per-stock aggregates, updated incrementally as packets are processed.
struct StockInfo {
    uint32_t stockID;
    uint8_t lastSellValue;
    uint8_t lastBuyValue;
    uint8_t minSellValue;
    uint8_t maxBuyValue;
    int64_t sumOrderValue;
    int orderCount;
    bool hasSellForMin;
    bool hasBuyForMax;

    StockInfo() : lastSellValue(0), lastBuyValue(0),
                  minSellValue(255), maxBuyValue(0),
                  sumOrderValue(0), orderCount(0),
                  hasSellForMin(false), hasBuyForMax(false) {}

    int getSpread() const {
        return abs((int)lastSellValue - (int)lastBuyValue);
    }
};

// Part - A
// Replays the order stream to build a running per-stock view, writing a
// sorted snapshot every `freq` orders (and a final one if the book doesn't
// divide evenly). The update loop itself is inherently sequential — each
// order can change what the *next* snapshot should contain — so only two
// independent pieces are parallelized:
//   1. Packet decoding, up front, since decoding order i doesn't depend on
//      decoding order i-1.
//   2. Each snapshot (copy state, sort, write file), dispatched as an OpenMP
//      task that runs in the background while the sequential loop moves on
//      to later orders — the copy is what makes this safe, since the task
//      then owns an independent, unchanging view of the book at that point.
void updateDisplay(const std::vector<uint64_t> &orderBook, int32_t freq) {
    int n = orderBook.size();

    // decode all packets in parallel — no data dependency between them
    std::vector<OrderBookEntry> all_entries(n);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        uint64_t unstuffed = unstuffBits(orderBook[i]);
        all_entries[i] = decodePacket(unstuffed);
    }

    // sequential replay: order i's effect on stockMap must be visible before
    // order i+1 is applied, so this loop itself is not parallelized
    int snapcnt = 0;
    std::unordered_map<uint32_t, StockInfo> stockMap;

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < n; i++) {
                const OrderBookEntry &entry = all_entries[i];
                StockInfo &stock = stockMap[entry.stockID];
                stock.stockID = entry.stockID;

                if (entry.orderType) {  // Sell
                    stock.lastSellValue = entry.orderValue;
                    stock.hasSellForMin = true;
                    if (entry.orderValue < stock.minSellValue) {
                        stock.minSellValue = entry.orderValue;
                    }
                } else {               // Buy
                    stock.lastBuyValue = entry.orderValue;
                    stock.hasBuyForMax = true;
                    if (entry.orderValue > stock.maxBuyValue) {
                        stock.maxBuyValue = entry.orderValue;
                    }
                }

                // every `freq` orders (or on the final, partial batch): snapshot
                if ((i + 1) % freq == 0 || (i == n - 1 && n % freq != 0)) {

                    // copy current state now, while it's still consistent —
                    // the background task will sort/write this copy while the
                    // sequential loop above keeps mutating the live stockMap
                    std::unordered_map<uint32_t, StockInfo> snapMap = stockMap;
                    int snapNo = snapcnt++;

                    #pragma omp task firstprivate(snapMap, snapNo)
                    {
                        std::vector<StockInfo> stocks;
                        stocks.reserve(snapMap.size());

                        for (auto &p : snapMap) {
                            stocks.push_back(p.second);
                        }

                        std::sort(stocks.begin(), stocks.end(),
                            [](const StockInfo &a, const StockInfo &b) {
                                int sa = a.getSpread();
                                int sb = b.getSpread();
                                if (sa != sb) return sa > sb;
                                return a.stockID > b.stockID;
                            });

                        std::string filename =
                            "snap_" + std::to_string(snapNo) + ".txt";
                        std::ofstream outFile(filename);

                        for (const auto &s : stocks) {
                            outFile << s.stockID << " "
                                    << (int)s.lastSellValue << " "
                                    << (int)s.lastBuyValue << " "
                                    << s.getSpread() << "\n";
                        }
                        outFile.close();
                    }
                }
            }

            // wait for all snapshot tasks , barrier
            #pragma omp taskwait
        }
    }
}


//Part-B
// Embarrassingly parallel: each order's contribution (qty * value) is
// independent of every other order, so a plain OpenMP reduction is safe and
// race-free without any locking.
int64_t totalAmountTraded(const std::vector<uint64_t> &orderBook) {
    int64_t total = 0;
    int n = orderBook.size();
    #pragma omp parallel for reduction(+:total)
    for (int i=0 ; i<n ; i++) {
        uint64_t unstuffed = unstuffBits(orderBook[i]);
        OrderBookEntry entry = decodePacket(unstuffed);
            
        total += (int64_t)entry.orderQty * (int64_t)entry.orderValue;
    }
    return total;
}

// Part - C
// Aggregates per-stock stats (min sell, max buy, average order value) across
// the whole order book. Unlike updateDisplay, order here doesn't matter — the
// final aggregate is the same regardless of processing order — so the whole
// pass is parallelized: each thread accumulates into its own thread-local
// map (avoiding lock contention on every single order), then the thread-local
// maps are merged into the shared result under one critical section at the end.
void printOrderStats(const std::vector<uint64_t> &orderBook) {
    int n = orderBook.size();

    // decode all packets in parallel
    std::vector<OrderBookEntry> all_entries(n);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        uint64_t unstuffed = unstuffBits(orderBook[i]);
        all_entries[i] = decodePacket(unstuffed);
    }

    // final merged per-stock aggregates
    std::unordered_map<uint32_t, StockInfo> stockMap;

    #pragma omp parallel
    {
        // thread-local map: avoids contending on a shared map for every order
        std::unordered_map<uint32_t, StockInfo> localMap;

        // order of processing doesn't affect the aggregate, so this loop is
        // safe to split across threads with no dependency between iterations
        #pragma omp for
        for (int i = 0; i < n; i++) {
            const OrderBookEntry &entry = all_entries[i];
            StockInfo &stock = localMap[entry.stockID];
            stock.stockID = entry.stockID;

            if (entry.orderType) {        // Sell
                stock.hasSellForMin = true;
                if (entry.orderValue < stock.minSellValue) {
                    stock.minSellValue = entry.orderValue;
                }
            } else {                       // Buy
                stock.hasBuyForMax = true;
                if (entry.orderValue > stock.maxBuyValue) {
                    stock.maxBuyValue = entry.orderValue;
                }
            }

            stock.sumOrderValue += (int64_t)entry.orderValue;
            stock.orderCount++;
        }

        // merge the local results
        #pragma omp critical
        {
            for (auto &p : localMap) {
                uint32_t id = p.first;
                StockInfo &global = stockMap[id];
                StockInfo &local  = p.second;

                global.stockID = id;
                global.sumOrderValue += local.sumOrderValue;
                global.orderCount   += local.orderCount;

                if (local.hasSellForMin) {
                    global.hasSellForMin = true;
                    global.minSellValue =
                        std::min(global.minSellValue, local.minSellValue);
                }

                if (local.hasBuyForMax) {
                    global.hasBuyForMax = true;
                    global.maxBuyValue =
                        std::max(global.maxBuyValue, local.maxBuyValue);
                }
            }
        }
    }

    //write the outputs 
    std::ofstream outFile("stats.txt");
    outFile << std::fixed << std::setprecision(4);

    for (auto &p : stockMap) {
        const StockInfo &stock = p.second;

        double avgOrderValue = 0.0;
        if (stock.orderCount > 0) {
            avgOrderValue =
                (double)stock.sumOrderValue / (double)stock.orderCount;
        }

        uint8_t minSell = stock.hasSellForMin ? stock.minSellValue : 0;
        uint8_t maxBuy  = stock.hasBuyForMax  ? stock.maxBuyValue  : 0;

        outFile << stock.stockID << " "
                << (int)minSell << " "
                << (int)maxBuy  << " "
                << std::fixed << std::setprecision(3)
                << avgOrderValue << "\n";
    }

    outFile.close();
}


/*
AI Usage Declaration:

An AI-based assistant was used during this assignment primarily for clarifying
conceptual doubts related to OpenMP, performance behavior, and interpretation
of assignment updates. The design decisions to speedup the sequential code were 
made .

The AI tool was used to understand certain parallel constructs and concepts
that were not covered in class at the time of the assignment (for example,
OpenMP tasking and the use of firstprivate), and to reason about observed
performance trends.

Example prompts used include:
- "Why is the speedup for updateDisplay limited?"
- "Would parallelizing packet decoding improve performance?"
- "Can reduction be used safely for summing values in OpenMP?"
- "Why does performance sometimes degrade with increasing thread count?"
- "Is using std::map acceptable given sparse stockID ranges?"
- "How does firstprivate work in OpenMP?"

*/
