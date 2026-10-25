--------------------------- MODULE Distributed ---------------------------
EXTENDS Integers, Sequences, FiniteSets, TLC
CONSTANTS Ranks, BudgetValues, TileValues, Rounds, Mutation
VARIABLES pc, localBudget, localTiles, trainable, host, plan, history, round, failed
vars == <<pc, localBudget, localTiles, trainable, host, plan, history, round, failed>>

Min(S) == CHOOSE x \in S : \A y \in S : x <= y
CommonBudget == IF \E r \in Ranks : host[r] THEN 1
                ELSE Min({localBudget[r] : r \in Ranks})
Terminal == \A r \in Ranks : pc[r] \in {"done", "error"}
Init == /\ pc = [r \in Ranks |-> "budget"]
        /\ localBudget \in [Ranks -> BudgetValues]
        /\ localTiles \in [Ranks -> TileValues]
        /\ trainable \in [Ranks -> BOOLEAN]
        /\ host \in [Ranks -> BOOLEAN]
        /\ plan = [r \in Ranks |-> 0]
        /\ history = [r \in Ranks |-> <<>>]
        /\ round = 0 /\ failed = FALSE

Agree == /\ \A r \in Ranks : pc[r] = "budget"
         /\ plan' = [r \in Ranks |-> IF Mutation = "localBudget"
                                   THEN localBudget[r] ELSE CommonBudget]
         /\ pc' = [r \in Ranks |-> "compute"]
         /\ UNCHANGED <<localBudget, localTiles, trainable, host, history, round, failed>>
Enter(r) == /\ pc[r] = "compute"
            /\ IF (Mutation = "skipEmpty" /\ localTiles[r] = 0)
                  \/ (Mutation = "skipUntrained" /\ ~trainable[r])
                  THEN /\ pc' = [pc EXCEPT ![r] = "done"] /\ UNCHANGED history
                  ELSE /\ pc' = [pc EXCEPT ![r] = "wait"]
                       /\ history' = [history EXCEPT ![r] = Append(@, <<round + 1, plan[r]>>)]
            /\ UNCHANGED <<localBudget, localTiles, trainable, host, plan, round, failed>>
Reduce == /\ \A r \in Ranks : pc[r] = "wait"
          /\ Cardinality({history[r][Len(history[r])] : r \in Ranks}) = 1
          /\ round' = round + 1
          /\ pc' = [r \in Ranks |-> IF round' = Rounds THEN "done" ELSE "compute"]
          /\ UNCHANGED <<localBudget, localTiles, trainable, host, plan, history, failed>>
Fault(r) == /\ ~failed /\ ~ (\E s \in Ranks : pc[s] = "done")
            /\ pc[r] \in {"budget", "compute", "wait"}
            /\ pc' = [pc EXCEPT ![r] = "error"] /\ failed' = TRUE
            /\ UNCHANGED <<localBudget, localTiles, trainable, host, plan, history, round>>
Abort == /\ failed /\ ~Terminal /\ Mutation # "localError"
         /\ pc' = [r \in Ranks |-> "error"]
         /\ UNCHANGED <<localBudget, localTiles, trainable, host, plan, history, round, failed>>
Next == Agree \/ Reduce \/ Abort \/ (\E r \in Ranks : Enter(r) \/ Fault(r))
        \/ (Terminal /\ UNCHANGED vars)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
BudgetAgreement == Cardinality({plan[r] : r \in Ranks}) = 1
Prefix(a, b) == Len(a) <= Len(b) /\ \A i \in 1..Len(a) : a[i] = b[i]
CollectiveOrder == \A r, s \in Ranks : Prefix(history[r], history[s]) \/ Prefix(history[s], history[r])
Termination == <>Terminal
ErrorTermination == failed ~> (\A r \in Ranks : pc[r] = "error")
RankSymmetry == Permutations(Ranks)
=============================================================================
