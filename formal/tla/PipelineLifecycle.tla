------------------------- MODULE PipelineLifecycle -------------------------
EXTENDS Integers, FiniteSets
CONSTANTS Jobs, Slots, Calls, Mutation
VARIABLES stage, hs, ds, host, device, hostUse, deviceUse, badRead, badLifetime, pc, pool, call
vars == <<stage, hs, ds, host, device, hostUse, deviceUse, badRead, badLifetime, pc, pool, call>>
Empty == [s \in Slots |-> 0]
Init == /\ stage = [j \in Jobs |-> 0] /\ hs = [j \in Jobs |-> CHOOSE s \in Slots : TRUE]
        /\ ds = hs /\ host = Empty /\ device = Empty /\ hostUse = Empty /\ deviceUse = Empty
        /\ badRead = FALSE /\ badLifetime = FALSE /\ pc = "run"
        /\ pool = Cardinality(Slots) /\ call = 1
Acquire(j, h, d) ==
  /\ pc = "run" /\ stage[j] = 0 /\ deviceUse[d] = 0
  /\ (hostUse[h] = 0 \/ Mutation = "reuseHost")
  /\ stage' = [stage EXCEPT ![j] = 1]
  /\ hs' = [hs EXCEPT ![j] = h] /\ ds' = [ds EXCEPT ![j] = d]
  /\ host' = [host EXCEPT ![h] = j]
  /\ hostUse' = [hostUse EXCEPT ![h] = j] /\ deviceUse' = [deviceUse EXCEPT ![d] = j]
  /\ UNCHANGED <<device, badRead, badLifetime, pc, pool, call>>
Upload(j) ==
  /\ stage[j] = 1 /\ stage' = [stage EXCEPT ![j] = 2]
  /\ device' = [device EXCEPT ![ds[j]] = host[hs[j]]]
  /\ hostUse' = [hostUse EXCEPT ![hs[j]] = 0]
  /\ badLifetime' = (badLifetime \/ host[hs[j]] # j)
  /\ UNCHANGED <<hs, ds, host, deviceUse, badRead, pc, pool, call>>
Compute(j) ==
  /\ pc = "run" /\ (stage[j] = 2 \/ (Mutation = "unfinishedUpload" /\ stage[j] = 1))
  /\ stage' = [stage EXCEPT ![j] = 3]
  /\ badRead' = (badRead \/ stage[j] # 2 \/ device[ds[j]] # j)
  /\ UNCHANGED <<hs, ds, host, device, hostUse, deviceUse, badLifetime, pc, pool, call>>
StartDownload(j, h) ==
  /\ pc = "run" /\ stage[j] = 3 /\ hostUse[h] = 0
  /\ hs' = [hs EXCEPT ![j] = h] /\ hostUse' = [hostUse EXCEPT ![h] = j]
  /\ stage' = [stage EXCEPT ![j] = 4]
  /\ UNCHANGED <<ds, host, device, deviceUse, badRead, badLifetime, pc, pool, call>>
Download(j) ==
  /\ stage[j] = 4 /\ stage' = [stage EXCEPT ![j] = 5]
  /\ host' = [host EXCEPT ![hs[j]] = device[ds[j]]]
  /\ deviceUse' = [deviceUse EXCEPT ![ds[j]] = 0]
  /\ UNCHANGED <<hs, ds, device, hostUse, badRead, badLifetime, pc, pool, call>>
Write(j) ==
  /\ pc = "run" /\ (stage[j] = 5 \/ (Mutation = "unfinishedDownload" /\ stage[j] = 4))
  /\ stage' = [stage EXCEPT ![j] = 6] /\ hostUse' = [hostUse EXCEPT ![hs[j]] = 0]
  /\ badRead' = (badRead \/ stage[j] # 5 \/ host[hs[j]] # j)
  /\ UNCHANGED <<hs, ds, host, device, deviceUse, badLifetime, pc, pool, call>>
Fail == /\ pc = "run" /\ pc' = "closing"
        /\ UNCHANGED <<stage, hs, ds, host, device, hostUse, deviceUse, badRead, badLifetime, pool, call>>
Close ==
  /\ (pc = "closing" \/ (pc = "run" /\ \A j \in Jobs : stage[j] = 6))
  /\ ((\A j \in Jobs : stage[j] \notin {1, 4}) \/ Mutation = "releaseEarly")
  /\ badLifetime' = (badLifetime \/ (\E j \in Jobs : stage[j] \in {1, 4}))
  /\ stage' = [j \in Jobs |-> 6] /\ pc' = "closed"
  /\ hostUse' = Empty /\ deviceUse' = Empty
  /\ pool' = IF Mutation = "leak" THEN pool ELSE 0
  /\ UNCHANGED <<hs, ds, host, device, badRead, call>>
Restart == /\ pc = "closed" /\ call < Calls /\ call' = call + 1
           /\ pc' = "run" /\ stage' = [j \in Jobs |-> 0]
           /\ pool' = Cardinality(Slots) + (IF Mutation = "growPool" THEN 1 ELSE 0)
           /\ UNCHANGED <<hs, ds, host, device, hostUse, deviceUse, badRead, badLifetime>>
Next == (\E j \in Jobs : (\E h, d \in Slots : Acquire(j, h, d))
          \/ Upload(j) \/ Compute(j) \/ (\E h \in Slots : StartDownload(j, h)) \/ Download(j) \/ Write(j))
        \/ Fail \/ Close \/ Restart \/ (pc = "closed" /\ call = Calls /\ UNCHANGED vars)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
NoUnfinishedRead == ~badRead
NoEarlyReuse == ~badLifetime
NoLeaks == pc = "closed" => pool = 0 /\ hostUse = Empty /\ deviceUse = Empty
PoolBound == pool <= Cardinality(Slots)
Termination == <>(pc = "closed" /\ call = Calls)
=============================================================================
