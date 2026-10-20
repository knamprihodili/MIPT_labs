#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>

class Dice {
public:
    virtual unsigned roll() = 0;
    virtual ~Dice() = default;
};

class SingleDice : public Dice {
public:
    SingleDice(unsigned max, unsigned seed) :
        max(max), dstr(1, max), reng(seed) { }

    unsigned roll() override {
        return dstr(reng);
    }

private:
    unsigned max;
    std::uniform_int_distribution<unsigned> dstr;
    std::mt19937 reng;
};

class ThreeDicePool : public Dice {
public:
    ThreeDicePool(Dice &d1, Dice &d2, Dice &d3) :
        dstr_1(d1), dstr_2(d2), dstr_3(d3) { }

    unsigned roll() override {
        return dstr_1.roll() + dstr_2.roll() + dstr_3.roll();
    }

private:
    Dice &dstr_1, &dstr_2, &dstr_3;
};

class PenaltyDice : public Dice {
public:
    PenaltyDice(Dice &d) : d(d) { }

    unsigned roll() override {
        unsigned r1 = d.roll();
        unsigned r2 = d.roll();
        return std::min(r1, r2);
    }

protected:
    Dice &d;
};

class BonusDice : public Dice {
public:
    BonusDice(Dice &d) : d(d) { }

    unsigned roll() override {
        unsigned r1 = d.roll();
        unsigned r2 = d.roll();
        return std::max(r1, r2);
    }

protected:
    Dice &d;
};

double expected_value(Dice &d, unsigned number_of_rolls = 1) {
    auto accum = 0llu;
    for (unsigned cnt = 0; cnt != number_of_rolls; ++cnt)
        accum += d.roll();
    return static_cast<double>(accum) / static_cast<double>(number_of_rolls);
}

double value_probability(unsigned value, Dice &d, unsigned number_of_rolls = 1) {
    unsigned hits = 0;
    for (unsigned cnt = 0; cnt != number_of_rolls; ++cnt) {
        if (d.roll() == value)
            ++hits;
    }
    return static_cast<double>(hits) / static_cast<double>(number_of_rolls);
}

int main() {
    const unsigned rolls = 100000;

    SingleDice d100(100, 42);
    PenaltyDice pen100(d100);
    BonusDice bon100(d100);

    std::ofstream out100("data/dice100.csv");
    out100 << "value,normal,penalty,bonus\n";
    for (unsigned v = 1; v <= 100; ++v) {
        out100 << v << ","
               << value_probability(v, d100, rolls) << ","
               << value_probability(v, pen100, rolls) << ","
               << value_probability(v, bon100, rolls) << "\n";
    }

    SingleDice d1(6, 1), d2(6, 2), d3(6, 3);
    ThreeDicePool pool(d1, d2, d3);
    PenaltyDice pen_pool(pool);
    BonusDice bon_pool(pool);

    std::ofstream out_pool("data/pool3d6.csv");
    out_pool << "value,normal,penalty,bonus\n";
    for (unsigned v = 1; v <= 18; ++v) {
        out_pool << v << ","
                 << value_probability(v, pool, rolls) << ","
                 << value_probability(v, pen_pool, rolls) << ","
                 << value_probability(v, bon_pool, rolls) << "\n";
    }

    return 0;
}