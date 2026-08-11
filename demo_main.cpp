// Demo driver for functions.cpp/functions.h — NOT part of the original assignment.
//
// The assignment only required submitting functions.cpp/functions.h; the
// course-provided packet feed/driver (which decoded raw exchange packets into
// orderBook) is intentionally not reproduced here. This file exists solely so
// the repo builds and runs standalone: it synthesizes a random order book and
// feeds it through the three OpenMP routines from functions.cpp.
//
// decodePacket() expects each uint64_t to carry a 49-bit payload
// (stockID:32 | orderType:1 | orderQty:8 | orderValue:8) that has already
// been bit-stuffed the same way updateDisplay/etc. undo via unstuffBits()
// (a run of 5 consecutive 1-bits is followed by one stuffed filler bit).
// stuffPacket() below is the inverse of that de-stuffing routine, derived by
// reading unstuffBits() in functions.cpp and verified to round-trip.

#include "functions.h"
#include <random>

static uint64_t stuffPacket(uint64_t value49) {
    uint64_t result = 0;
    int outPos = 0;
    int consecutiveOnes = 0;
    for (int srcPos = 0; srcPos < 49; srcPos++) {
        uint64_t bit = (value49 >> srcPos) & 1ULL;
        result |= (bit << outPos);
        outPos++;
        if (bit) {
            if (++consecutiveOnes == 5) {
                consecutiveOnes = 0;
                outPos++;  // insert stuffed filler bit
            }
        } else {
            consecutiveOnes = 0;
        }
    }
    return result;
}

static uint64_t makePacket(uint32_t stockID, bool isSell, uint8_t qty, uint8_t value) {
    uint64_t raw = (uint64_t)stockID
                 | ((uint64_t)(isSell ? 1 : 0) << 32)
                 | ((uint64_t)qty << 33)
                 | ((uint64_t)value << 41);
    return stuffPacket(raw);
}

int main() {
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> stockDist(1, 20);
    std::uniform_int_distribution<int> sideDist(0, 1);
    std::uniform_int_distribution<int> qtyDist(1, 100);
    std::uniform_int_distribution<int> valueDist(1, 200);

    const int N = 5000;
    std::vector<uint64_t> orderBook;
    orderBook.reserve(N);
    for (int i = 0; i < N; i++) {
        orderBook.push_back(makePacket(
            stockDist(rng), sideDist(rng) == 1, (uint8_t)qtyDist(rng), (uint8_t)valueDist(rng)));
    }

    std::cout << "Generated " << N << " synthetic order-book packets across "
              << "20 stocks.\n";

    int64_t total = totalAmountTraded(orderBook);
    std::cout << "Total amount traded: " << total << "\n";

    printOrderStats(orderBook);
    std::cout << "Per-stock stats written to stats.txt\n";

    updateDisplay(orderBook, /*freq=*/1000);
    std::cout << "Snapshots written to snap_*.txt\n";

    return 0;
}
