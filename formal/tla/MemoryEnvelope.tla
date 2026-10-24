-------------------------- MODULE MemoryEnvelope --------------------------
EXTENDS Integers
CONSTANTS
  \* @type: Int;
  DeviceBudget,
  \* @type: Int;
  HostBudget,
  \* @type: Int;
  MaxCost,
  \* @type: Int;
  Calls,
  \* @type: Str;
  Mutation
VARIABLES
  \* @type: Int;
  pc,
  \* @type: Int;
  work,
  \* @type: Int;
  copies,
  \* @type: Int;
  graph,
  \* @type: Int;
  tiles,
  \* @type: Int;
  host,
  \* @type: Int;
  call
\* @type: <<Int, Int, Int, Int, Int, Int, Int>>;
vars == <<pc, work, copies, graph, tiles, host, call>>
Init == /\ pc = 0 /\ work = 0 /\ copies = 0 /\ graph = 0
        /\ tiles = 0 /\ host = 0 /\ call = 1
Request(w, c, g, t, h, next) ==
  LET estimated == (IF Mutation = "work" THEN 0 ELSE w)
                   + (IF Mutation = "copy" THEN 0 ELSE c)
                   + (IF Mutation = "graph" THEN 0 ELSE g)
                   + (IF Mutation = "reduction" THEN 0 ELSE t)
      fits == estimated <= DeviceBudget /\ (Mutation = "host" \/ h <= HostBudget)
  IN /\ IF fits THEN /\ work' = w /\ copies' = c /\ graph' = g
                     /\ tiles' = t /\ host' = h /\ pc' = next
                ELSE /\ pc' = 6 /\ UNCHANGED <<work, copies, graph, tiles, host>>
     /\ UNCHANGED call
Release == /\ pc = 6 /\ pc' = 7
           /\ work' = IF Mutation = "leak" THEN work ELSE 0
           /\ copies' = 0 /\ graph' = 0 /\ tiles' = 0 /\ host' = 0
           /\ UNCHANGED call
Next == (pc = 0 /\ \E n \in 1..MaxCost : Request(n, 0, 0, 0, 0, 1))
     \/ (pc = 1 /\ \E n \in 1..MaxCost : Request(work, n, 0, 0, 0, 2))
     \/ (pc = 2 /\ \E n \in 1..MaxCost : Request(work, copies, n, 0, 0, 3))
     \/ (pc = 3 /\ Request(work, copies, graph,
                          IF Mutation = "twoTiles" THEN 2 ELSE 1, 0, 4))
     \/ (pc = 4 /\ \E n \in 1..MaxCost : Request(work, copies, graph, tiles, n, 5))
     \/ (pc = 5 /\ pc' = 6 /\ UNCHANGED <<work, copies, graph, tiles, host, call>>)
     \/ Release
     \/ (pc = 7 /\ call < Calls /\ call' = call + 1 /\ pc' = 0
         /\ UNCHANGED <<work, copies, graph, tiles, host>>)
     \/ (pc = 7 /\ call = Calls /\ UNCHANGED vars)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
DeviceBound == work + copies + graph + tiles <= DeviceBudget
HostBound == host <= HostBudget
OneReducedTile == tiles <= 1
NoLeaks == pc = 7 => work + copies + graph + tiles + host = 0
Termination == <>(pc = 7 /\ call = Calls)
=============================================================================
