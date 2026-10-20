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

class PenaltyDice : virtual public Dice {
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

class BonusDice : virtual public Dice {
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

class DoubleDice : public PenaltyDice, public BonusDice {
public:
    DoubleDice(Dice &d) : PenaltyDice(d), BonusDice(d) { }

    unsigned roll() override {
        return PenaltyDice::roll() + BonusDice::roll();
    }
};

class DoubleDiceComp : public Dice {
public:
    DoubleDiceComp(Dice &d) : pen(d), bon(d) { }

    unsigned roll() override {
        return pen.roll() + bon.roll();
    }

private:
    PenaltyDice pen;
    BonusDice bon;
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
    DoubleDice dd(d100);
    DoubleDiceComp dd_comp(d100);

    std::cout << "DoubleDice expected value: " << expected_value(dd, rolls) << '\n';
    std::cout << "DoubleDiceComp expected value: " << expected_value(dd_comp, rolls) << '\n';

    std::ofstream out("data/double_dice.csv");
    out << "value,prob,prob_comp\n";
    for (unsigned v = 2; v <= 200; ++v) {
        out << v << ","
            << value_probability(v, dd, rolls) << ","
            << value_probability(v, dd_comp, rolls) << "\n";
    }

    return 0;
}