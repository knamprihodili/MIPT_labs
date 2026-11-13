#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <functional>
#include <iterator>
#include <algorithm>
#include <unordered_set>
#include <chrono>
#include <cassert>

template <typename T, typename Compare = std::less<T>>
class SkipList {
private:
    struct Node {
        T value;
        std::vector<Node*> next;
        Node* prev;

        Node(const T& val = T(), int level = 1)
            : value(val), next(level, nullptr), prev(nullptr) {}
    };

    Node* head;
    Node* tail;
    int max_level;
    size_t elem_count;
    double p;
    Compare cmp;
    std::mt19937 rng;

    // метод инициализации для всех конструкторов
    void init() {
        max_level = 1;
        elem_count = 0;
        head = new Node();  
        tail = new Node();
        head->next[0] = tail;
        tail->prev = head;
    }

    // функция для вероятностей
    int random_level() {
        int lvl = 1;
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        while (lvl <= max_level && dist(rng) < p) {
            lvl++;
        }
        return lvl;
    }

    // меняем местами все внутренние переменные двух списков (своего и другого) (нужно для перемещения за О(1))
    void swap(SkipList& other) {
        std::swap(head, other.head);
        std::swap(tail, other.tail);
        std::swap(max_level, other.max_level);
        std::swap(elem_count, other.elem_count);
        std::swap(p, other.p);
        std::swap(cmp, other.cmp);
    }

public:

    // двунаправленный итератор
    class iterator {
    private:
        Node* current;
        friend class SkipList;

    public:
        // позволяют стандартным 
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T*;
        using reference = const T&;

        iterator(Node* node = nullptr) : current(node) {}

        // перегруженные операторы разыменования
        const T& operator*() const { return current->value; }
        const T* operator->() const { return &(current->value); }

        // перегруженные операторы шага (с постфиксными версиями)
        iterator& operator++() {
            current = current->next[0];
            return *this;
        }
        
        iterator operator++(int) {
            iterator temp = *this;
            current = current->next[0];
            return temp;
        }
        
        iterator& operator--() {
            current = current->prev;
            return *this;
        }

        iterator operator--(int) {
            iterator temp = *this;
            current = current->prev;
            return temp;
        }

        // перегруженные операторы сравнения
        bool operator==(const iterator& other) const { return current == other.current; }
        bool operator!=(const iterator& other) const { return current != other.current; }
    };

    // обратный итератор
    using reverse_iterator = std::reverse_iterator<iterator>;

    // конструктор по умолчанию
    SkipList() : p(0.5), cmp(Compare()), rng(std::random_device{}()) {
        init();
    }

    explicit SkipList(Compare comp, double prob = 0.5)
        : p(prob), cmp(comp), rng(std::random_device{}()) {
        init();
    }

    // конструктор от диапазона итераторов
    template <typename InputIt>
    SkipList(InputIt first, InputIt last, Compare comp = Compare(), double prob = 0.5)
        : p(prob), cmp(comp), rng(std::random_device{}()) {
        init();
        insert(first, last);
    }

    // конструктор копирования и оператор (копирующего) присваивания
    SkipList(const SkipList& other)
        : p(other.p), cmp(other.cmp), rng(std::random_device{}()) {
        init();
        insert(other.begin(), other.end());
    }

    SkipList& operator=(const SkipList& other) {
        if (this != &other) {
            clear();
            cmp = other.cmp;
            p = other.p;
            insert(other.begin(), other.end());
        }
        return *this;
    }

    // перемещающий конструктор и оператор перемещающего присваивания (передача управения теми же ячейками памяти без копирования)
    SkipList(SkipList&& other)
        : p(0.5), cmp(Compare()), rng(std::random_device{}()) {
        init();
        swap(other);
    }

    SkipList& operator=(SkipList&& other) {
        if (this != &other) {
            clear();
            swap(other);
        }
        return *this;
    }

    // деструктор
    ~SkipList() {
        clear();
        delete head;
        delete tail;
    }

    // проверка пустоты и размера
    bool empty() const { return elem_count == 0; }
    size_t size() const { return elem_count; }

    iterator begin() const { return iterator(head->next[0]); }
    iterator end() const { return iterator(tail); }
    reverse_iterator rbegin() const { return reverse_iterator(end()); }
    reverse_iterator rend() const { return reverse_iterator(begin()); }

    // обычная вставка
    // находим место на каждом уровне, выбираем случайную высоту и перенаправляем ссылки через новый узел
    iterator insert(const T& val) {
        std::vector<Node*> update(max_level, head);
        Node* cur = head;

        for (int i = max_level - 1; i >= 0; --i) {
            while (cur->next[i] != tail && !cmp(val, cur->next[i]->value)) {
                cur = cur->next[i];
            }
            update[i] = cur;
        }

        int new_level = random_level();
        if (new_level > max_level) {
            for (int i = max_level; i < new_level; ++i) {
                head->next.push_back(tail);
                update.push_back(head);
            }
            max_level = new_level;
        }

        Node* new_node = new Node(val, new_level);
        for (int i = 0; i < new_level; ++i) {
            new_node->next[i] = update[i]->next[i];
            update[i]->next[i] = new_node;
        }

        new_node->prev = update[0];
        new_node->next[0]->prev = new_node;
        elem_count++;

        return iterator(new_node);
    }

    // перегруженный insert для диапазона
    template <typename InputIt>
    void insert(InputIt first, InputIt last) {
        for (InputIt it = first; it != last; ++it) {
            insert(*it);
        }
    }

    // возвращает первый элемент не меньше заданного
    iterator lower_bound(const T& key) const {
        Node* cur = head;
        for (int i = max_level - 1; i >= 0; --i) {
            while (cur->next[i] != tail && cmp(cur->next[i]->value, key)) {
                cur = cur->next[i];
            }
        }
        return iterator(cur->next[0]);
    }

    // возвращает первый элемент строго больше заданного
    iterator upper_bound(const T& key) const {
        Node* cur = head;
        for (int i = max_level - 1; i >= 0; --i) {
            while (cur->next[i] != tail && !cmp(key, cur->next[i]->value)) {
                cur = cur->next[i];
            }
        }
        return iterator(cur->next[0]);
    }

    // поиск, возвращающий первый из одинаковых
    iterator find(const T& key) const {
        iterator it = lower_bound(key);
        if (it != end() && !cmp(key, *it)) {
            return it;
        }
        return end();
    }

    // возвращает промежуток итераторов на равные заданному ключу элементы
    std::pair<iterator, iterator> equal_range(const T& key) const {
        return {lower_bound(key), upper_bound(key)};
    }

    // возвращает количество вхождений повторяющегося элемента
    size_t count(const T& key) const {
        auto range = equal_range(key);
        return std::distance(range.first, range.second);
    }

    // очистка, использующаяся в деструкторах и при копировании, чтобы не было утечек памяти
    void clear() {
        Node* cur = head->next[0];
        while (cur != tail) {
            Node* next_node = cur->next[0];
            delete cur;
            cur = next_node;
        }
        head->next.assign(1, tail);
        tail->prev = head;
        max_level = 1;
        elem_count = 0;
    }

    // удаление элемента из списка по итератору
    iterator erase(iterator pos) {
        Node* target = pos.current;
        Node* next_elem = target->next[0];
        std::vector<Node*> update(max_level, head);
        Node* cur = head;

        for (int i = max_level - 1; i >= 0; --i) {
            while (cur->next[i] != tail && cmp(cur->next[i]->value, target->value)) {
                cur = cur->next[i];
            }
            update[i] = cur;
        }

        for (size_t i = 0; i < target->next.size(); ++i) {
            Node* p = update[i];
            while (p->next[i] != tail && p->next[i] != target) {
                p = p->next[i];
            }
            if (p->next[i] == target) {
                p->next[i] = target->next[i];
            }
        }

        target->next[0]->prev = target->prev;
        delete target;
        elem_count--;

        return iterator(next_elem);
    }

    // перегруженный erase
    iterator erase(iterator first, iterator last) {
        while (first != last) {
            first = erase(first);
        }
        return last;
    }
};

void benchmark(size_t n, double prob, std::ofstream& csv) {
    const int repeats = 5;
    double ins_sum = 0, fnd_sum = 0, ers_sum = 0;

    for (int r = 0; r < repeats; ++r) {
        std::vector<int> data(n);
        for (size_t i = 0; i < n; ++i) data[i] = i;
        std::shuffle(data.begin(), data.end(), std::mt19937(42 + r));

        SkipList<int> sl(std::less<int>(), prob);
        std::vector<SkipList<int>::iterator> iters(n);

        auto t1 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < n; ++i) sl.insert(data[i]);
        auto t2 = std::chrono::steady_clock::now();

        for (size_t i = 0; i < n; ++i) iters[i] = sl.find(data[i]);
        auto t3 = std::chrono::steady_clock::now();

        for (size_t i = 0; i < n; ++i) sl.erase(iters[i]);
        auto t4 = std::chrono::steady_clock::now();

        ins_sum += std::chrono::duration<double, std::nano>(t2 - t1).count() / n;
        fnd_sum += std::chrono::duration<double, std::nano>(t3 - t2).count() / n;
        ers_sum += std::chrono::duration<double, std::nano>(t4 - t3).count() / n;
    }

    double ins = ins_sum / repeats;
    double fnd = fnd_sum / repeats;
    double ers = ers_sum / repeats;

    std::cout << "N = " << n << ", p = " << prob
              << " | insert: " << ins << " ns, find: " << fnd
              << " ns, erase: " << ers << " ns\n";
    csv << n << "," << prob << "," << ins << "," << fnd << "," << ers << "\n";
}

int main() {
    std::unordered_set<int> us = {5, 1, 4, 2, 3};
    SkipList<int> list(us.begin(), us.end());
    list.insert(3);
    list.insert(3);

    assert(!list.empty() && list.size() == 7);
    assert(list.count(3) == 3 && list.find(99) == list.end());

    auto it1 = list.find(1);
    list.erase(list.find(3));
    assert(list.count(3) == 2 && *it1 == 1);

    auto range = list.equal_range(3);
    list.erase(range.first, range.second);
    assert(list.count(3) == 0 && *list.rbegin() == 5);

    SkipList<int> copy_list = list;
    SkipList<int> moved_list = std::move(copy_list);
    assert(moved_list.size() == 4 && copy_list.empty());

    std::cout << "Tests OK\n\n";

    std::ofstream csv_n("bench_n.csv");
    csv_n << "n,p,insert_ns,find_ns,erase_ns\n";
    for (size_t n : {1000, 2500, 5000, 10000, 25000, 50000, 100000, 200000, 350000, 500000}) {
        benchmark(n, 0.5, csv_n);
    }

    std::cout << "\n";
    std::ofstream csv_p("bench_p.csv");
    csv_p << "n,p,insert_ns,find_ns,erase_ns\n";
    for (double p : {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95}) {
        benchmark(100000, p, csv_p);
    }

    return 0;
}