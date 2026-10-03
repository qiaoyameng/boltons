from pytest import mark, raises

from boltons.setutils import IndexedSet, _MISSING, complement


def test_indexed_set_basic():
    zero2nine = IndexedSet(range(10))
    five2nine = zero2nine & IndexedSet(range(5, 15))
    x = IndexedSet(five2nine)
    x |= {10}

    assert list(zero2nine) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert set(zero2nine) == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    assert list(five2nine) == [5, 6, 7, 8, 9]
    assert x == IndexedSet([5, 6, 7, 8, 9, 10])
    assert x[-1] == 10

    assert zero2nine ^ five2nine == IndexedSet([0, 1, 2, 3, 4])

    assert x[:3] == IndexedSet([5, 6, 7])
    assert x[2:4:-1] == IndexedSet([8, 7])


def test_indexed_set_rsub():
    "From #252"
    assert (set('abc') - IndexedSet('bcd')) == {'a'}
    assert (IndexedSet('abc') - IndexedSet('bcd')) == IndexedSet(['a'])
    assert (frozenset('abc') - IndexedSet('bcd')) == frozenset(['a'])


@mark.parametrize('iterable_type', [list, tuple, iter])
def test_indexed_set_update_multiple_iterables(iterable_type):
    items = IndexedSet([1])

    result = items.update(iterable_type([2, 1, 3]), iterable_type([]),
                          iterable_type([3, 4, 2]))

    assert result is None
    assert list(items) == [1, 2, 3, 4]


def test_indexed_set_update_multiple_iterables_with_tuple_items():
    items = IndexedSet()

    items.update([(1, 2)], [(3, 4), (1, 2)])

    assert list(items) == [(1, 2), (3, 4)]


def test_indexed_set_mutate():
    thou = IndexedSet(range(1000))
    assert (thou.pop(), thou.pop()) == (999, 998)
    assert (thou.pop(499), thou.pop(499)) == (499, 500)

    ref = [495, 496, 497, 498, 501, 502, 503, 504, 505, 506]
    assert [thou[i] for i in range(495, 505)] == ref

    assert len(thou) == 996
    while len(thou) > 600:
        dead_idx_len = len(thou.dead_indices)
        dead_idx_count = thou._dead_index_count
        thou.pop(0)
        new_dead_idx_len = len(thou.dead_indices)
        if new_dead_idx_len < dead_idx_len:
            assert dead_idx_count > 0
            # 124, 109, 95
    assert len(thou) == 600
    assert thou._dead_index_count == 67

    assert not any([thou[i] is _MISSING for i in range(len(thou))])

    thou &= IndexedSet(range(500, 503))

    assert thou == IndexedSet([501, 502])
    return


def big_popper():
    # more of a benchmark than a test
    from os import urandom
    import time
    big_set = IndexedSet(range(100000))
    rands = [ord(r) for r in urandom(len(big_set))]
    start_time, start_size = time.time(), len(big_set)
    while len(big_set) > 10000:
        if len(big_set) % 10000 == 0:
            print(len(big_set) / 10000)
        rand = rands.pop()
        big_set.pop(rand)
        big_set.pop(-rand)
    end_time, end_size = time.time(), len(big_set)
    print()
    print('popped {} items in {} seconds'.format(start_size - end_size,
                                             end_time - start_time))


def test_complement_set():
    '''exercises a bunch of code-paths but doesn't really confirm math identities'''
    assert complement(complement(set())) == set()
    sab = set('ab')
    sbc = set('bc')
    cab = complement('ab')
    cbc = complement('bc')
    cc = complement('c')
    sc = set('c')
    u = complement(set())
    assert repr(sab) in repr(cab)
    # non-mutating tests
    assert cab != cbc
    assert complement(cab) == sab
    assert complement(cbc) == sbc
    assert 'a' not in cab
    assert 'c' in cab
    assert (sab & cbc) == (sab - sbc)  # set theory invariant
    assert not (cab < sab)  # complement never subset of set
    assert not (sab < cab)
    assert not (cbc < sab)
    assert not (cbc < cab)  # not subsets of each other
    assert sab < cc
    assert cab < (cab | cbc)
    assert (cab | cbc) > cab
    assert cc > sab
    assert not (cab > sab)
    # <= / >= between two complement sets (comp(X) <= comp(Y) iff Y subset X)
    assert cab <= complement('a')
    assert not (complement('a') <= cab)
    assert cab <= cab
    assert complement('a') >= cab
    assert cab >= cab
    # double-complemented (included-form) sets dispatch to the + +
    # branches of __le__/__ge__: comp(comp(X)) behaves like X
    ccab = complement(complement(set('ab')))
    cca = complement(complement(set('a')))
    assert ccab >= cca
    assert not (cca >= ccab)
    assert cca <= ccab
    assert not (ccab <= cca)
    assert ccab >= ccab
    assert ccab <= ccab
    assert not cab.isdisjoint(cc)  # complements never disjoint
    assert cab.isdisjoint(sab)
    assert not cab.isdisjoint(sc)
    assert (cab | sab) == u
    assert (cab | cc) == u
    assert (cab | cbc) == complement('b')
    assert (sab | cab) == (cbc | sbc)
    assert (sab & cab) == (cbc & sbc)
    assert (sab ^ cab) == (cbc ^ sbc)
    assert cab - cc == sc
    assert cab - sab == cab
    assert sab - cab == sab
    assert (cab ^ cbc | set('b')) == (sab | sbc)
    everything = complement(frozenset())
    assert everything in everything  # https://en.wikipedia.org/wiki/Russell%27s_paradox
    assert bool(cab)
    assert not complement(u)
    # destructive testing
    cab ^= sab
    cab ^= sab
    cab &= sab
    cab &= cbc
    cab |= sab
    cab |= cbc
    cab -= sab
    cab.add(5)
    cab.remove(5)
    cab.update(sab)
    cab.discard(sab)
    cab.update(cbc)
    cab.add(complement(frozenset()))  # frozen complement can be a member of complement set
    assert len({complement(frozenset()): 1, complement(frozenset()): 2}) == 1  # hash works
    with raises(NotImplementedError): cab.pop()
    with raises(NotImplementedError): len(cab)
    with raises(NotImplementedError): iter(cab)
    ~cab
    cab.complement()
    cab.complemented()
    class OpOverloader:
        # tests that operators properly return NotImplemented so they will defer to
        # another class implementation if available
        def __and__(self, other): return self
        __rand__ = __iand__ = __or__ = __ror__ = __ior__ = __xor__ = __rxor__ = __sub__ = __isub__ = __and__
        def __le__(self, other): return True
        __lt__ = __ge__ = __gt__ = __le__

    ops = OpOverloader()
    def opsmash(a, b):
        a &= b; a |= b; a -= b; a ^= b
        a > b; a >= b; a < b; a <= b
        return (((a & b) | b) - b) ^ b

    with raises(TypeError): opsmash(cab, object())
    assert opsmash(ops, cab) == ops
    assert opsmash(cab, ops) == ops


def test_iset_index_method():
    original_list = list(range(8, 20)) + list(range(8))

    indexed_list = IndexedSet()

    for i in original_list:
        indexed_list.add(i)

    for i in original_list:
        index = indexed_list.index(i)
        # if we're removing them in order, the target value should always be at index 0
        assert index == 0
        indexed_list.pop(index)

    indexed_list = IndexedSet(range(10))

    for i in reversed(range(10)):
        if i % 2:
            continue
        index = indexed_list.index(i)
        assert i == indexed_list.pop(index)


    indexed_list = IndexedSet(range(32))

    for i in list(indexed_list):
        if i % 3:
            index = indexed_list.index(i)
            assert i == indexed_list.pop(index)

    indexed_list = IndexedSet(range(10))

    for i in range(10):
        if i < 3:
            continue
        index = indexed_list.index(i)
        assert i == indexed_list.pop(index)

    indexed_list = IndexedSet(range(32))

    for i in list(indexed_list):
        if i % 3:
            index = indexed_list.index(i)
            assert i == indexed_list.pop(index)


def test_iset_slice_after_removal():
    # PR #423: slice bounds were mapped into item_list space while islice
    # consumed the apparent (dead-slot-free) stream, so any removal made
    # slices over-count.
    x = IndexedSet(range(10))
    x.pop(2)  # leaves [0, 1, 3, 4, 5, 6, 7, 8, 9] with a dead slot
    assert list(x[1:4]) == [1, 3, 4]
    assert list(x[-3:]) == [7, 8, 9]
    assert list(x[2:-1]) == [3, 4, 5, 6, 7, 8]
    assert isinstance(x[1:4], IndexedSet)


def test_iset_slice_agreement():
    # Slicing must not depend on dead slots: a set with removals must slice
    # identically to a freshly-built set with the same contents (all steps,
    # including the reversed-window negative-step semantics), and positive
    # steps must match list slicing exactly.
    bounds = (None, -12, -9, -5, -1, 0, 1, 4, 8, 9, 12)
    steps = (None, 1, 2, 3, -1, -2, -3)
    for pops in ([], [2], [0, 1, 2], [7, 8, 9], [0, 3, 6, 9], [1, 2, 5, 6]):
        iset = IndexedSet(range(10))
        for p in pops:
            iset.discard(p)
        fresh = IndexedSet(iset)  # same contents, no dead slots
        ref = list(iset)
        for start in bounds:
            for stop in bounds:
                for step in steps:
                    s = slice(start, stop, step)
                    assert list(iset[s]) == list(fresh[s]), (pops, s)
                    if step is None or step > 0:
                        assert list(iset[s]) == ref[s], (pops, s)


def test_iset_scalar_index_bounds():
    # Out-of-range negatives used to wrap silently: __getitem__ normalized
    # the index and _get_real_index normalized it AGAIN, so x[-10] on a
    # 9-element set returned x[-1], and pop(-15) removed a middle element.
    x = IndexedSet(range(10))
    x.pop(2)  # len 9, one dead slot
    assert x[-1] == 9
    assert x[-9] == 0
    with raises(IndexError):
        x[9]
    with raises(IndexError):
        x[-10]
    with raises(IndexError):
        x.pop(-15)
    assert 4 in x  # pop(-15) formerly silently removed 4
    with raises(IndexError):
        IndexedSet()[0]


def test_iset_scalar_index_multiple_dead_intervals():
    # exercise the _get_real_index interval walk across several dead ranges
    iset = IndexedSet(range(100))
    for p in (5, 20, 40, 60, 80):
        iset.discard(p)
    assert len(iset.dead_indices) == 5  # below compaction threshold (100/8)
    ref = list(iset)
    for i in list(range(-95, 95, 7)) + [0, -1, 94, -95]:
        assert iset[i] == ref[i]
    with raises(IndexError):
        iset[95]
    with raises(IndexError):
        iset[-96]


def test_bulk_difference_update():
    # removals large enough to take the O(n) bulk path (> _MAX_DEAD_INTERVALS)
    thing = IndexedSet(range(1000))
    thing.difference_update(set(range(0, 1000, 2)))
    assert list(thing) == list(range(1, 1000, 2))
    assert thing[0] == 1 and thing[-1] == 999
    assert thing.index(501) == 250
    thing.add(0)
    assert thing[-1] == 0


def test_bulk_intersection_update():
    thing = IndexedSet(range(1000))
    thing.intersection_update(set(range(500, 2000)))
    assert list(thing) == list(range(500, 1000))
    assert thing[0] == 500 and thing.index(750) == 250


def test_small_difference_update_unchanged():
    # small removals keep the incremental path
    thing = IndexedSet(range(100))
    thing.difference_update({1, 3, 5})
    assert list(thing)[:5] == [0, 2, 4, 6, 7]


def _ref_move(seq, start, stop, destination):
    # plain-list reference for IndexedSet.move_range semantics: slice-style
    # clipping, pre-move destination boundary, self/empty range is a no-op
    items = list(seq)
    n = len(items)

    def clip(value):
        if value < 0:
            value += n
        return min(max(value, 0), n)

    start, stop, destination = clip(start), clip(stop), clip(destination)
    if start >= stop or start <= destination <= stop:
        return items
    moved = items[start:stop]
    if destination < start:
        return (items[:destination] + moved
                + items[destination:start] + items[stop:])
    return (items[:start] + items[stop:destination]
            + moved + items[destination:])


def test_iset_move_range_basic():
    iset = IndexedSet(range(6))

    assert iset.move_range(1, 3, 4) is None  # [1, 2] before 4
    assert list(iset) == [0, 3, 1, 2, 4, 5]

    assert iset.move_range(2, 4, 0) is None  # move to the front
    assert list(iset) == [1, 2, 0, 3, 4, 5]

    assert iset.move_range(0, 2, 6) is None  # move to the end
    assert list(iset) == [0, 3, 4, 5, 1, 2]


def test_iset_move_range_boundaries_and_empty():
    original = list(range(6))

    # destination at/inside the moved range, including both adjacent
    # boundaries, and empty ranges are all no-ops
    for start, stop, destination in ((2, 4, 2), (2, 4, 3), (2, 4, 4),
                                     (0, 0, 3), (3, 3, 0), (4, 2, 1)):
        iset = IndexedSet(original)
        assert iset.move_range(start, stop, destination) is None
        assert list(iset) == original

    # destination just beyond the range actually moves
    iset = IndexedSet(original)
    iset.move_range(1, 3, 3)
    assert list(iset) == original  # == stop, no-op
    iset.move_range(1, 3, 4)
    assert list(iset) == [0, 3, 1, 2, 4, 5]

    # negative indices
    iset = IndexedSet(range(6))
    iset.move_range(0, 2, -1)
    assert list(iset) == [2, 3, 4, 0, 1, 5]
    iset = IndexedSet(range(6))
    iset.move_range(-2, 6, 0)
    assert list(iset) == [4, 5, 0, 1, 2, 3]

    # empty set
    empty = IndexedSet()
    assert empty.move_range(0, 0, 0) is None
    assert empty.move_range(-5, 50, 2) is None
    assert list(empty) == []


def test_iset_move_range_out_of_range_clipping():
    # bounds clip like list slices instead of raising
    iset = IndexedSet(range(6))
    iset.move_range(-100, 2, 3)
    assert list(iset) == [2, 0, 1, 3, 4, 5]

    iset = IndexedSet(range(6))
    iset.move_range(4, 100, 0)
    assert list(iset) == [4, 5, 0, 1, 2, 3]

    iset = IndexedSet(range(6))
    iset.move_range(0, 2, 100)  # clips to the end boundary
    assert list(iset) == [2, 3, 4, 5, 0, 1]

    iset = IndexedSet(range(6))
    iset.move_range(4, 6, -100)  # clips to the front boundary
    assert list(iset) == [4, 5, 0, 1, 2, 3]

    with raises(TypeError):
        IndexedSet(range(6)).move_range(1.5, 3, 0)


@mark.parametrize('removed', ([], [50], [10, 45], [5, 25, 50]))
def test_iset_move_range_matches_list_reference(removed):
    bounds = (-100, -70, -9, -3, -1, 0, 1, 2, 5, 8, 9, 50, 63, 64, 100)
    pops = removed
    seq = [x for x in range(64) if x not in set(pops)]
    for start in bounds:
        for stop in bounds:
            for destination in bounds:
                iset = IndexedSet(range(64))
                for value in pops:
                    iset.discard(value)
                had_dead_slots = bool(iset.dead_indices)

                iset.move_range(start, stop, destination)
                expected = _ref_move(seq, start, stop, destination)
                assert list(iset) == expected, (pops, start, stop,
                                                destination)
                if expected != seq:
                    # actual move: holes are compacted regardless of
                    # pre-existing dead slots
                    assert iset.dead_indices == []
                    assert iset._dead_index_count == 0
                if had_dead_slots:
                    # same result as a dead-slot-free copy moved identically
                    fresh = IndexedSet(seq)
                    fresh.move_range(start, stop, destination)
                    assert list(fresh) == expected


def test_iset_move_range_preserves_identity_and_membership():
    items = [object() for _ in range(6)]
    iset = IndexedSet(items)
    before = list(iset)

    iset.pop(2)  # leave a dead slot before moving
    iset.move_range(0, 2, 3)

    after = list(iset)
    expected = [before[3], before[0], before[1], before[4], before[5]]
    assert all(got is want for got, want in zip(after, expected))
    assert {id(x) for x in after} == {id(x) for x in before if x is not
                                      before[2]}
    assert len(iset) == len(after) == len(set(after))


def test_iset_move_range_after_heavy_deletion():
    iset = IndexedSet(range(1000))
    ref = list(range(1000))

    for value in range(0, 600, 2):  # many individual pops force compactions
        iset.remove(value)
    ref = [value for value in ref if value not in range(0, 600, 2)]
    assert iset._compactions >= 1

    iset.move_range(100, 300, 0)
    ref = _ref_move(ref, 100, 300, 0)
    assert list(iset) == ref
    assert iset.dead_indices == []


def test_iset_move_range_mixed_operations():
    # chained index/pop/slice/reversed/update sequence must stay in
    # lockstep with the equivalent operations on a plain list
    iset = IndexedSet(range(24))
    ref = list(range(24))

    for value in (3, 7, 11):
        iset.remove(value)
        ref.remove(value)
    assert iset.dead_indices  # holes present

    iset.move_range(2, 8, 0)
    ref = _ref_move(ref, 2, 8, 0)
    assert list(iset) == ref

    assert iset.index(ref[4]) == ref.index(ref[4])
    assert iset.pop(1) == ref.pop(1)
    assert list(iset[2:6]) == ref[2:6]
    assert list(iset[-4:]) == ref[-4:]
    assert list(reversed(iset)) == list(reversed(ref))

    iset.update([99, ref[0], 5])
    for value in (99, ref[0], 5):
        if value not in ref:
            ref.append(value)
    assert list(iset) == ref

    iset.move_range(0, 3, -1)
    ref = _ref_move(ref, 0, 3, -1)
    assert list(iset) == ref
    assert set(iset) == set(ref)
    assert list(iset[::-1]) == ref[::-1]
