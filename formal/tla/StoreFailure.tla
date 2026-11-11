------------------------------ MODULE StoreFailure ------------------------------
EXTENDS Integers, FiniteSets
CONSTANTS Profile, Mutation
VARIABLES epoch, round, pc, remaining, errors, reported, stats,
          faultKind, faultRound, faultRead, failedRequest

Ranks == 1..4
Groups == {{1, 2}, {3, 4}}
Jobs(r) == IF r <= 2 THEN 2 ELSE IF Profile = "empty" THEN 0 ELSE 1
Reads(r) == IF r % 2 = 1 THEN 2 ELSE 1
ReadWork(r, k) == IF Jobs(r) >= k THEN Reads(r) ELSE 0
vars == <<epoch, round, pc, remaining, errors, reported, stats,
          faultKind, faultRound, faultRead, failedRequest>>
FixedFault == <<faultKind, faultRound, faultRead>>
Failed == \E r \in Ranks : errors[r] = "original"
AllAt(label) == \A r \in Ranks : pc[r] = label
Terminal == \A r \in Ranks : pc[r] \in {"error", "done"}

Init == /\ epoch = 0 /\ round = 1
        /\ pc = [r \in Ranks |-> "read"]
        /\ remaining = [r \in Ranks |-> ReadWork(r, 1)]
        /\ errors = [r \in Ranks |-> "none"]
        /\ reported = [r \in Ranks |-> "none"]
        /\ stats = [r \in Ranks |-> FALSE]
        /\ faultKind \in {"read", "write", "flush"}
        /\ faultRound \in 1..2 /\ faultRead \in 1..2
        /\ failedRequest = FALSE

\* One read action is one externally failing store access, not a whole loop.
Read(r) == /\ pc[r] = "read" /\ remaining[r] > 0
           /\ LET fail == epoch = 0 /\ r = 1 /\ faultKind = "read"
                          /\ round = faultRound /\ remaining[r] = Reads(r) - faultRead + 1
              IN /\ remaining' = [remaining EXCEPT ![r] = @ - 1]
                 /\ errors' = [errors EXCEPT ![r] = IF fail THEN "original" ELSE @]
                 /\ pc' = [pc EXCEPT ![r] = IF fail THEN
                          IF Mutation = "skipReadAgreement" THEN "error" ELSE "readGate" ELSE @]
                 /\ reported' = [reported EXCEPT ![r] =
                          IF fail /\ Mutation = "skipReadAgreement" THEN "original" ELSE @]
           /\ UNCHANGED <<epoch, round, stats, FixedFault, failedRequest>>
ReadDone(r) == /\ pc[r] = "read" /\ remaining[r] = 0
               /\ pc' = [pc EXCEPT ![r] = "readGate"]
               /\ UNCHANGED <<epoch, round, remaining, errors, reported, stats,
                              FixedFault, failedRequest>>

\* All parent ranks reach this boundary, including padded/empty jobs.
AgreeRead == /\ AllAt("readGate") /\ Mutation # "skipReadAgreement"
             /\ pc' = [r \in Ranks |-> IF Failed THEN "error" ELSE "reduce"]
             /\ reported' = [r \in Ranks |-> IF Failed THEN "original" ELSE "none"]
             /\ failedRequest' = (failedRequest \/ Failed)
             /\ UNCHANGED <<epoch, round, remaining, errors, stats, FixedFault>>
SkipReadAgreement(r) == /\ Mutation = "skipReadAgreement" /\ pc[r] = "readGate"
                        /\ pc' = [pc EXCEPT ![r] = "reduce"]
                        /\ UNCHANGED <<epoch, round, remaining, errors, reported, stats,
                                       FixedFault, failedRequest>>

\* Subgroups may finish their numerical SUM independently. No parent gate
\* appears inside the unequal per-rank read traversal above.
Reduce(group) == /\ \A r \in group : pc[r] = "reduce"
                 /\ \A r \in group : Jobs(r) >= round
                 /\ pc' = [r \in Ranks |-> IF r \in group THEN "write" ELSE pc[r]]
                 /\ UNCHANGED <<epoch, round, remaining, errors, reported, stats,
                                FixedFault, failedRequest>>
\* An exhausted subgroup has no numerical SUM; it advances its padded job.
SkipEmptyGroup(group) == /\ \A r \in group : pc[r] = "reduce" /\ Jobs(r) < round
                         /\ pc' = [r \in Ranks |-> IF r \in group THEN "write" ELSE pc[r]]
                         /\ UNCHANGED <<epoch, round, remaining, errors, reported, stats,
                                        FixedFault, failedRequest>>
Write(r) == /\ pc[r] = "write"
            /\ LET fail == epoch = 0 /\ r = 1 /\ faultKind = "write"
                           /\ round = faultRound /\ Jobs(r) >= round
               IN errors' = [errors EXCEPT ![r] = IF fail THEN "original" ELSE @]
            /\ pc' = [pc EXCEPT ![r] = "writeGate"]
            /\ UNCHANGED <<epoch, round, remaining, reported, stats, FixedFault, failedRequest>>
AgreeWrite == /\ AllAt("writeGate")
              /\ pc' = [r \in Ranks |-> IF Failed THEN "error" ELSE
                                       IF round = 2 THEN "flush" ELSE "read"]
              /\ reported' = [r \in Ranks |-> IF Failed THEN "original" ELSE "none"]
              /\ remaining' = [r \in Ranks |-> IF Failed \/ round = 2 THEN 0
                                              ELSE ReadWork(r, round + 1)]
              /\ round' = IF Failed \/ round = 2 THEN round ELSE round + 1
              /\ failedRequest' = (failedRequest \/ Failed)
              /\ UNCHANGED <<epoch, errors, stats, FixedFault>>
Flush(r) == /\ pc[r] = "flush"
            /\ LET fail == epoch = 0 /\ r = 1 /\ faultKind = "flush"
               IN errors' = [errors EXCEPT ![r] = IF fail THEN "original" ELSE @]
            /\ pc' = [pc EXCEPT ![r] = "flushGate"]
            /\ UNCHANGED <<epoch, round, remaining, reported, stats, FixedFault, failedRequest>>
AgreeFlush == /\ AllAt("flushGate") /\ Mutation # "skipFlushAgreement"
              /\ pc' = [r \in Ranks |-> IF Failed THEN "error" ELSE "done"]
              /\ reported' = [r \in Ranks |-> IF Failed THEN "original" ELSE "success"]
              /\ stats' = [r \in Ranks |-> ~Failed]
              /\ failedRequest' = (failedRequest \/ Failed)
              /\ UNCHANGED <<epoch, round, remaining, errors, FixedFault>>
SkipFlushAgreement(r) == /\ Mutation = "skipFlushAgreement" /\ pc[r] = "flushGate"
                         /\ pc' = [pc EXCEPT ![r] = IF errors[r] = "original" THEN "error" ELSE "done"]
                         /\ reported' = [reported EXCEPT ![r] =
                                            IF errors[r] = "original" THEN "original" ELSE "success"]
                         /\ stats' = [stats EXCEPT ![r] = errors[r] # "original"]
                         /\ UNCHANGED <<epoch, round, remaining, errors, FixedFault, failedRequest>>

\* Second admission uses the same parent/subgroups. Errors belong to a call;
\* retaining its flag is the progress negative control.
Retry == /\ epoch = 0 /\ AllAt("error")
         /\ epoch' = 1 /\ round' = 1
         /\ pc' = [r \in Ranks |-> "read"]
         /\ remaining' = [r \in Ranks |-> ReadWork(r, 1)]
         /\ errors' = IF Mutation = "retainError" THEN errors ELSE [r \in Ranks |-> "none"]
         /\ reported' = [r \in Ranks |-> "none"]
         /\ stats' = [r \in Ranks |-> FALSE]
         /\ failedRequest' = TRUE
         /\ UNCHANGED FixedFault
Next == (\E r \in Ranks : Read(r) \/ ReadDone(r) \/ SkipReadAgreement(r) \/ Write(r) \/ Flush(r)
                         \/ SkipFlushAgreement(r))
        \/ AgreeRead \/ (\E group \in Groups : Reduce(group) \/ SkipEmptyGroup(group))
        \/ AgreeWrite \/ AgreeFlush \/ Retry
        \/ (epoch = 1 /\ Terminal /\ UNCHANGED vars)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

TypeOK == /\ epoch \in 0..1 /\ round \in 1..2
          /\ pc \in [Ranks -> {"read", "readGate", "reduce", "write", "writeGate",
                               "flush", "flushGate", "error", "done"}]
          /\ remaining \in [Ranks -> 0..2]
          /\ errors \in [Ranks -> {"none", "original"}]
          /\ reported \in [Ranks -> {"none", "original", "success"}]
          /\ stats \in [Ranks -> BOOLEAN] /\ failedRequest \in BOOLEAN
          /\ faultKind \in {"read", "write", "flush"}
          /\ faultRound \in 1..2 /\ faultRead \in 1..2
NoOrphanedReduction == \A group \in Groups :
    (\E r \in group : pc[r] = "reduce") => (\A r \in group : pc[r] # "error")
ErrorAgreement == Terminal /\ (\E r \in Ranks : reported[r] = "original") =>
                  \A r \in Ranks : reported[r] = "original"
NoSuccessfulStatsOnFailure == (Failed \/ (\E r \in Ranks : reported[r] = "original")) =>
                              \A r \in Ranks : ~stats[r]
HealthyRetry == (\E r \in Ranks : stats[r]) => epoch = 1 /\ failedRequest
Recovery == <>(epoch = 1 /\ AllAt("done"))
NotFailedLaterRead == ~(epoch = 0 /\ faultKind = "read" /\ faultRead = 2 /\ Failed)
NotEmptyParticipant == ~(Jobs(3) = 0 /\ pc[3] = "readGate" /\ pc[1] = "read")
NotRecovered == ~(epoch = 1 /\ AllAt("done") /\ failedRequest)
NotFlushFailure == ~(epoch = 0 /\ faultKind = "flush" /\ Failed)
=============================================================================
