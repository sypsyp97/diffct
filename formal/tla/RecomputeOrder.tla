-------------------------- MODULE RecomputeOrder --------------------------
EXTENDS Integers, Sequences
CONSTANTS Ranks, Tiles, Rays, Pieces, Parameters, Mutation
VARIABLES index, history, project, replicated, reduceInput
vars == <<index, history, project, replicated, reduceInput>>
Pre(p) == IF replicated THEN (IF p THEN Rays ELSE Tiles) ELSE 0
Body(p) == Tiles * Rays + (IF reduceInput THEN (IF p THEN Tiles ELSE Rays) ELSE 0)
Count == Pre(project) + Pieces * Body(project) + Parameters
Event(i, p) ==
  IF i <= Pre(p) THEN <<"output", 0, IF p THEN 0 ELSE i, IF p THEN i ELSE 0>>
  ELSE IF i > Pre(p) + Pieces * Body(p)
       THEN <<"parameter", i - Pre(p) - Pieces * Body(p), 0, 0>>
  ELSE LET n == i - Pre(p) - 1
           piece == (n \div Body(p)) + 1
           width == (IF p THEN Rays ELSE Tiles) + (IF reduceInput THEN 1 ELSE 0)
           outer == ((n % Body(p)) \div width) + 1
           inner == (n % Body(p)) % width
       IN IF reduceInput /\ inner = 0
          THEN <<"input", piece, IF p THEN outer ELSE 0, IF p THEN 0 ELSE outer>>
          ELSE <<"pair", piece, IF p THEN outer ELSE inner + (IF reduceInput THEN 0 ELSE 1),
                               IF p THEN inner + (IF reduceInput THEN 0 ELSE 1) ELSE outer>>
Init == /\ index = [r \in Ranks |-> 0] /\ history = [r \in Ranks |-> <<>>]
        /\ project \in BOOLEAN /\ replicated \in BOOLEAN /\ reduceInput \in BOOLEAN
Step(r) == /\ index[r] < Count
           /\ index' = [index EXCEPT ![r] = @ + 1]
           /\ history' = [history EXCEPT ![r] = Append(@, Event(index[r] + 1,
                             IF Mutation = "rankOrder" /\ r = (CHOOSE s \in Ranks : TRUE)
                             THEN ~project ELSE project))]
           /\ UNCHANGED <<project, replicated, reduceInput>>
Done == \A r \in Ranks : index[r] = Count
Next == (\E r \in Ranks : Step(r)) \/ (Done /\ UNCHANGED vars)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
Expected == [i \in 1..Count |-> Event(i, project)]
ForwardOrder == \A r \in Ranks : history[r] = SubSeq(Expected, 1, Len(history[r]))
Termination == <>Done
=============================================================================
