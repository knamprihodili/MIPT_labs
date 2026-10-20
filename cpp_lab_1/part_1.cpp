#include <iostream>
#include <random>

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
    Dice & dstr_1, &dstr_2, &dstr_3;
};

double expected_value(Dice &d, unsigned number_of_rolls = 1) {
    auto accum = 0llu;
    for (unsigned cnt = 0; cnt != number_of_rolls; ++cnt)
        accum += d.roll();
    return static_cast<double>(accum) / static_cast<double>(number_of_rolls);
}

int main() {
    SingleDice d(6, 1);

    SingleDice d1(6, 1), d2(6, 2), d3(6, 3);
    ThreeDicePool tdp(d1, d2, d3);

    std::cout << "Dice: " << expected_value(d, 100000) << '\n';
    std::cout << "ThreeDicePool: " << expected_value(tdp, 100000) << '\n';

    return 0;
}