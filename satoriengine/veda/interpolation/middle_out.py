def middle_out(j: int, bias: bool = False) -> list[int]:
    """
    Return a list of all integers [1..j-1] in the 'middle-out' order, row by row,
    using a balanced BST BFS plus the sub-list splitting strategy described.

    :param j: Upper bound; we produce the ordering of 1..j-1.
    :param bias:
        False => for an even-length sub-list, pick the lower middle index first
        True  => for an even-length sub-list, pick the upper middle index first
    """
    if j <= 1:
        return []

    # ----------------------------------------------------------------
    # 1) Build a balanced BST of [1..(j-1)]
    # ----------------------------------------------------------------
    class Node:
        def __init__(self, val):
            self.val = val
            self.left = None
            self.right = None

    def build_balanced_bst(sorted_vals):
        """Recursively build a balanced BST from a sorted list."""
        if not sorted_vals:
            return None
        mid = len(sorted_vals) // 2
        root = Node(sorted_vals[mid])
        root.left = build_balanced_bst(sorted_vals[:mid])
        root.right = build_balanced_bst(sorted_vals[mid+1:])
        return root

    values = list(range(1, j))  # [1..j-1]
    root = build_balanced_bst(values)

    # ----------------------------------------------------------------
    # 2) BFS to gather levels (rows) of the BST
    # ----------------------------------------------------------------
    from collections import deque
    queue = deque([root])
    levels = []
    while queue:
        level_size = len(queue)
        row = []
        for _ in range(level_size):
            node = queue.popleft()
            if node:
                row.append(node.val)
                queue.append(node.left)
                queue.append(node.right)
        if row:
            levels.append(row)

    # ----------------------------------------------------------------
    # 3) A helper to pick from a single BFS "row" in the desired fashion
    # ----------------------------------------------------------------
    def pick_from_level(arr, bias_flag):
        """
        Given a list of integers (one BFS row), pick them in "middle-out" order:
          1) Always pick from the largest sub-list first.
          2) Among sub-lists of the same size, pick the one whose center is
             closest to the row's overall center (abs difference).
          3) 'Middle' index for each sub-list depends on bias_flag:
               - bias_flag=False => pick the 'lower-middle' if even length
               - bias_flag=True  => pick the 'upper-middle' if even length
          4) Removing that element splits the sub-list into left & right; re-insert them.
          5) Continue until arr is exhausted.
        """
        if not arr:
            return []

        # Precompute the row's overall center => average
        row_center = sum(arr) / len(arr)

        # We'll keep a list of sub-lists, each storing:
        #   (list_of_values, creation_id, sum_of_values)
        # We store sum_of_values to quickly compute the sub-list's center for tie-breaking.
        sublists = []
        creation_counter = 0

        # Start with one sub-list = the entire row
        sublists.append((arr, creation_counter, sum(arr)))
        creation_counter += 1

        result = []

        def get_middle_index(n, is_upper_bias):
            """Compute which index is the 'middle' for a list of length n."""
            if n == 1:
                return 0
            if (n % 2) == 1:
                # odd length => single exact middle
                return n // 2
            else:
                # even length => depends on bias
                if is_upper_bias:
                    return n // 2
                else:
                    return (n // 2) - 1

        import math

        def sublist_center(subvals, sum_sub):
            """Return the average of subvals ( = sum_sub / len(subvals) )."""
            return sum_sub / len(subvals)

        # We'll define a function to re-sort sublists each time we add new ones.
        # Sort priority:
        #   (1) size descending => -len(subvals)
        #   (2) distance from row_center ascending => abs(center - row_center)
        #   (3) creation_id ascending => stable insertion
        def sort_sublists():
            sublists.sort(
                key=lambda x: (
                    -len(x[0]),  # bigger first
                    abs((x[2] / len(x[0])) - row_center),  # closer to row_center
                    x[1]  # lower creation_id => earlier
                )
            )

        def insert_sublist(subvals):
            nonlocal creation_counter
            if not subvals:
                return
            s = sum(subvals)
            sublists.append((subvals, creation_counter, s))
            creation_counter += 1

        # Each time we pick, we:
        #   1) sort sublists with the above criteria
        #   2) take the first sub-list
        #   3) pick that sub-list’s ‘middle’ element
        #   4) split into left/right
        #   5) re-insert left & right
        #   6) repeat

        while sublists:
            # 1) sort
            sort_sublists()
            # 2) take first
            big_sub, cid, sum_sub = sublists.pop(0)
            n = len(big_sub)
            if n == 0:
                continue
            # 3) find middle index
            m_idx = get_middle_index(n, bias_flag)
            m_val = big_sub[m_idx]
            result.append(m_val)
            # 4) split
            left_part = big_sub[:m_idx]
            right_part = big_sub[m_idx+1:]
            # 5) re-insert
            if left_part:
                insert_sublist(left_part)
            if right_part:
                insert_sublist(right_part)

        return result

    # ----------------------------------------------------------------
    # 4) For each BFS row, pick out the "middle-out" sequence
    # ----------------------------------------------------------------
    final_order = []
    for row in levels:
        row_result = pick_from_level(row, bias)
        final_order.extend(row_result)

    return final_order


# ----------------------------------------------------------------------
# Demonstration
if __name__ == "__main__":
    print("j=16 =>", middle_out(16, bias=True))
    print("j=16 =>", middle_out(16, bias=False))
    # Expected: [8, 4, 12, 6, 10, 2, 14, 7, 11, 3, 13, 9, 5, 15, 1]

    print("j=32 =>", middle_out(32, bias=False))
    print("j=10 =>", middle_out(10, bias=False))
    print("j=10 =>", middle_out(10, bias=True))
    #print("j=2000 =>", middle_out(2000, bias=False))
    # Now in the row [2,6,10,14,18,22,26,30],
    # we pick 14, then 22, then among [2,6,10] & [18], [26,30],
    # we next pick from the largest => [2,6,10] => pick 6, then [26,30] => pick 26,
    # then the new sub-lists are single-element => [2], [10], [30], [18].
    # Among those same-size sub-lists, we choose them in order of closeness to 16:
    #    [18] (distance=2), [10](6), [2](14), [30](14).
    # => so 10 is chosen before 2.
    # That should fix the problem you reported.
