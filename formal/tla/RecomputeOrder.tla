-------------------------- MODULE RecomputeOrder --------------------------
EXTENDS Integers, Sequences
CONSTANTS Ranks, Tiles, Rays, Pieces, Mutation
VARIABLES index, history, project
vars == <<index, history, project>>
Count == Tiles * Rays * Pieces
Pair(i, p) == LET n == (i - 1) % (Tiles * Rays)
             IN <<((i - 1) \div (Tiles * Rays)) + 1,
                  IF p THEN (n \div Rays) + 1 ELSE (n % Tiles) + 1,
                  IF p THEN (n % Rays) + 1 ELSE (n \div Tiles) + 1>>
Init == /\ index = [r \in Ranks |-> 0] /\ history = [r \in Ranks |-> <<>>]
        /\ project \in BOOLEAN
Step(r) == /\ index[r] < Count
           /\ index' = [index EXCEPT ![r] = @ + 1]
           /\ history' = [history EXCEPT ![r] = Append(@, Pair(index[r] + 1,
                             IF Mutation = "rankOrder" /\ r = (CHOOSE s \in Ranks : TRUE)
                             THEN ~project ELSE project))]
           /\ UNCHANGED project
Done == \A r \in Ranks : index[r] = Count
Next == (\E r \in Ranks : Step(r)) \/ (Done /\ UNCHANGED vars)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
Expected == [i \in 1..Count |-> Pair(i, project)]
ForwardOrder == \A r \in Ranks : history[r] = SubSeq(Expected, 1, Len(history[r]))
Termination == <>Done
=============================================================================
