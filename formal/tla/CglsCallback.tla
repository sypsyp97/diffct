---------------------------- MODULE CglsCallback ----------------------------
EXTENDS Naturals, FiniteSets
CONSTANTS Ranks, MaxIterations, Mutation
VARIABLES phase, iteration, mode, returnsFalse, pending, changed, callbackErrors,
          stopRequested, valid, checkpointValid, badNumerics
vars == <<phase, iteration, mode, returnsFalse, pending, changed, callbackErrors,
          stopRequested, valid, checkpointValid, badNumerics>>

\* Numerical vectors are abstracted to whether recursive r agrees with b-Ax.
\* Only callback-supplied x is modeled. External operator/global mutation and
\* mutate-then-restore byte ABA are outside a before/after fingerprint contract.
\* Checkpoints publish BEFORE callbacks; a failed callback cannot corrupt them.
Init == /\ phase = "ready"
        /\ iteration = 0
        /\ mode \in [Ranks -> {"absent", "observer", "mutator", "raises"}]
        /\ returnsFalse \in [Ranks -> BOOLEAN]
        /\ pending = {} /\ changed = {} /\ callbackErrors = {}
        /\ stopRequested = FALSE
        /\ valid = [r \in Ranks |-> TRUE]
        /\ checkpointValid = TRUE /\ badNumerics = FALSE

Numerical == /\ phase = "ready" /\ iteration < MaxIterations
             /\ badNumerics' = (badNumerics \/ \E r \in Ranks : ~valid[r])
             /\ checkpointValid' = \A r \in Ranks : valid[r]
             /\ iteration' = iteration + 1 /\ phase' = "callbacks"
             /\ pending' = Ranks /\ changed' = {} /\ callbackErrors' = {}
             /\ stopRequested' = FALSE
             /\ UNCHANGED <<mode, returnsFalse, valid>>

Callback(r) == /\ phase = "callbacks" /\ r \in pending
               /\ pending' = pending \ {r}
               /\ phase' = IF pending = {r} THEN "gate" ELSE phase
               /\ changed' = IF mode[r] = "mutator" THEN changed \cup {r} ELSE changed
               /\ valid' = IF mode[r] = "mutator"
                            THEN [valid EXCEPT ![r] = FALSE] ELSE valid
               /\ callbackErrors' = IF mode[r] = "raises"
                                      THEN callbackErrors \cup {r} ELSE callbackErrors
               /\ stopRequested' = (stopRequested \/
                    (mode[r] \in {"observer", "mutator"} /\ returnsFalse[r]))
               /\ UNCHANGED <<iteration, mode, returnsFalse, checkpointValid, badNumerics>>

Detected == callbackErrors # {} \/
            (Mutation # "legacy" /\ changed # {})
Gate == /\ phase = "gate" /\ pending = {}
        /\ phase' = IF Mutation = "stopFirst" /\ stopRequested THEN "done"
                    ELSE IF Mutation = "localGate" THEN
                      \* A rank without a callback ignores a peer's failure.
                      IF \E r \in Ranks : mode[r] = "absent"
                      THEN (IF stopRequested \/ iteration = MaxIterations THEN "done" ELSE "ready")
                      ELSE (IF Detected THEN "error"
                            ELSE IF stopRequested \/ iteration = MaxIterations THEN "done" ELSE "ready")
                    ELSE IF Detected THEN "error"
                    ELSE IF stopRequested \/ iteration = MaxIterations THEN "done" ELSE "ready"
        /\ UNCHANGED <<iteration, mode, returnsFalse, pending, changed, callbackErrors,
                       stopRequested, valid, checkpointValid, badNumerics>>

Advance == Numerical \/ (\E r \in Ranks : Callback(r)) \/ Gate
Terminal == /\ phase \in {"done", "error"} /\ UNCHANGED vars
Next == Advance \/ Terminal
\* Assumptions: each synchronous callback returns or raises; control collectives
\* complete; the enabled next solver action is scheduled. No claim for hung Python.
Spec == Init /\ [][Next]_vars /\ WF_vars(Advance)
UnfairSpec == Init /\ [][Next]_vars

TypeOK == /\ phase \in {"ready", "callbacks", "gate", "done", "error"}
          /\ iteration \in 0..MaxIterations
          /\ mode \in [Ranks -> {"absent", "observer", "mutator", "raises"}]
          /\ returnsFalse \in [Ranks -> BOOLEAN]
          /\ pending \subseteq Ranks /\ changed \subseteq Ranks /\ callbackErrors \subseteq Ranks
          /\ stopRequested \in BOOLEAN /\ valid \in [Ranks -> BOOLEAN]
          /\ checkpointValid \in BOOLEAN /\ badNumerics \in BOOLEAN
NoStaleNumerics == ~badNumerics
MutationErrorPrecedence == phase = "done" => changed = {}
CollectiveErrorGate == phase \in {"ready", "done"} =>
                        (changed = {} /\ callbackErrors = {})
CheckpointCoherent == checkpointValid
Termination == <>(phase \in {"done", "error"})
MutationStopNotReached == ~(phase = "error" /\ changed # {} /\ stopRequested /\ pending = {})
ObserverDoneNotReached == ~(phase = "done" /\ iteration = MaxIterations /\
                            \A r \in Ranks : mode[r] = "observer" /\ ~returnsFalse[r])
=============================================================================
