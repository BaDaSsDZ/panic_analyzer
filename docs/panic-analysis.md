# CASI AI — Panic Data Analysis

Analysis of 4 real production panics to understand what data is available and how it maps to tags.

---

## Key findings

### What signals actually matter for tagging

After reviewing all 4 panics, the meaningful signals are:

| Signal | Value | Notes |
|---|---|---|
| `creation_method` | Always `4` (occurrence/API) or `0` (alerter) | Low signal on its own — most are API-triggered Bolt/ride panics |
| `is_false_alarm` | bool | Direct signal for `#AccidentalPress` / false alarm tags |
| `panic_logs` | Rich timeline | Most useful: log `type` integers (not the generic descriptions) |
| `panic_comments` | Free text | **Highest signal** — operators narrate exactly what happened |
| `services` dispatched | Armed Response, Private Medical | Tells us the severity/nature |
| `dynamic_form_answers` | "Incident Description" field | Free text from responder on scene |
| `special_procedures` | e.g. "Private Ambulance IN-TRIP" | Context about alerter type (Bolt driver etc.) |
| `occurrence.type` | `1` = Ride | Confirms Bolt/ride context |

### What is NOT useful for tagging

- `comment_alerter` / `comment_responder` — almost always null
- `panic_type` — null on all 4 panics
- `tier` — all normal tier, not discriminating
- Location change logs (type=32) — pure noise, dozens of entries, carry no semantic meaning
- Observer escalation logs (type=60) — operational noise, already filtered

### Critical discovery: log type 32 (Significant Location Change) is noise

Panic 286926 had **60+ location change logs** — all identical pattern, zero tagging value.
**Must be excluded from training input.**

### Log types that carry signal (keep these)

| Type | Meaning |
|---|---|
| 17 | Incident Created |
| 4 | Incident Acknowledged |
| 38 | Service assigned |
| 2 | Partner dispatched |
| 19 | Service accepted (responder en route) |
| 6 | Responder arrived on scene |
| 7 | Time on scene recorded |
| 21 | Service/incident completed |
| 18 | Manually resolved |
| 15 | False alarm confirmed |
| 25 | Calling alerter |
| 26 | Contact made with alerter |
| 27 | No answer from alerter |
| 31 | Comment created (use actual comment text instead) |
| 39 | ETA updated |
| 66 | Auto-dispatch stopped |
| 74 | Dispatch sent/received |
| 20 | Incident declined |

**Exclude:** type 32 (location noise), type 60 (observer escalation noise), type null (generic admin ack)

---

## Panic-by-panic breakdown

---

### Panic 286965 — Bolt driver accidental press

**Core:** creation_method=4 (occurrence), is_false_alarm=false, status=complete  
**Occurrence:** type=1 (Ride), creation_method=api  
**Services:** Armed Response (Manually Resolved)  
**Duration:** ~23 minutes  

**Key log events:**
- Incident Created → Acknowledged → Armed Response assigned → Lapua dispatched
- Responder accepted, en route 11min ETA
- Arrived on scene (07h07m response time)
- Calling Alerter → Completed

**Key comments:**
- "Bolt driver pressed the panic not answering" ← `#CallNotAnswered`
- "No vehicle currently available"
- "Armed response made contact with the driver, who confirmed was accidental. The number on the app is no longer in use" ← `#AccidentalPress`, `#ROContactMade`

**Dynamic form (responder):**
- Incident Description: "Bolt driver press panic but he said that he has a problem with the phone or gps"

**Special procedures:** "Private Ambulance IN-TRIP" (Bolt/ride context)

**Ground truth tags:**
- `#CallNotAnswered` — driven by: comment "not answering" + log type 25 "Calling Alerter"
- `#ArrivedAtLocation` — driven by: log type 6 "arrived on scene"
- `#ROContactMade` — driven by: comment "made contact with driver"
- `#AccidentalPress` — driven by: comment "confirmed was accidental" + form "he said he has a problem"

---

### Panic 286926 — Bolt driver vehicle accident, medical emergency

**Core:** creation_method=4 (occurrence), is_false_alarm=false, status=complete  
**Occurrence:** type=1 (Ride), creation_method=api  
**Services:** Armed Response + Private Medical (Rocket HEMS)  
**Duration:** ~2h45min (complex incident)  

**Key log events:**
- Incident Created → Acknowledged → Private Medical assigned → Rocket HEMS dispatched
- Armed Response added → Benneth Security accepted
- Both responders arrive on scene
- Manually Resolved

**Key comments:**
- "bolt driver involved in an accident while on a trip please dispatch private ambulance and bill to CASI passenger is injured" ← `#VehicleAccident`
- "Nare Olvies Malema Client: +27631187691 bolt driver got accident the passenger is injured"
- "Just spoke to the client, ambulance is on scene."
- "spoke to bolt driver... he confirmed Medical emergency arrived at 12h06"
- "Ride Vehicle Details, Make: Suzuki, Model: S-Presso, Registration: MP27MDGP"
- "Driver called the police he will give the reference"
- "Bolt driver got accident the passenger is injured dispatched rocket, he called the police"

**Dynamic form (responder):**
- Incident Description: "Accident"

**Special procedures:** "Private Ambulance IN-TRIP" (Bolt/ride context)

**Ground truth tags:**
- `#PanicAlarm` — generic tag, panic was real
- `#VehicleAccident` — driven by: comments "got accident", "passenger is injured", "dispatch ambulance"
- `#ArrivedAtLocation` — driven by: log type 6 arrived on scene
- `#ROContactMade` — driven by: comment "spoke to bolt driver...confirmed"

---

### Panic 286916 — Bolt driver accidental press (false alarm confirmed)

**Core:** creation_method=4 (occurrence), is_false_alarm=**true**, status=complete  
**Occurrence:** type=1 (Ride), creation_method=api  
**Services:** Armed Response (Manually Resolved)  
**Duration:** ~21 minutes  

**Key log events:**
- Incident Created → Acknowledged → TrekT Log dispatched
- Cannot reach client → responder en route
- Responder arrived on scene
- **Log type 15: "Panic Emergency State set to False Alarm"** ← direct signal
- Completed

**Key comments:**
- "eta 15 mins"
- "@Trek T please assist, cant get hold of the client" ← `#CallNotAnswered`
- "the wife says everything is fine we can cancel" ← family confirmed safe
- "vehicle is not at the location tho"
- "vehicle located"
- "all in order with the client"
- "accidental press" ← `#AccidentalPress` (responder confirmed on scene)

**Dynamic form (responder):**
- Incident Description: "bolt" (minimal)

**Ground truth tags:**
- `#CallNotAnswered` — driven by: comment "cant get hold of client"
- `#ArrivedAtLocation` — driven by: log type 6 arrived on scene
- `#ROContactMade` — driven by: comment "all in order with the client", "vehicle located"
- `#AccidentalPress` — driven by: comment "accidental press" + log type 15 false alarm + is_false_alarm=true

---

### Panic 286655 — Suspicious vehicle, alerter-triggered

**Core:** creation_method=**0** (alerter — user pressed button), is_false_alarm=false, status=complete  
**Occurrence:** null (not a ride/occurrence panic)  
**Services:** Armed Response (RO On Scene)  
**Duration:** ~45 minutes  

**Key log events:**
- Incident Created → Acknowledged → Calling Alerter immediately
- Armed Response assigned → Mvumi Responder accepted fast (auto-dispatch worked)
- Responder arrived on scene
- Manually Resolved
- Incident Rated (alerter gave positive feedback)

**Key comments:**
- "Client requesting an armed response at carishma funeral" ← location context
- "Armed response confirmed... vehicle in the yard without a registration looked suspicious. All in order and contact made with client" ← `#FeelingUnsafe`, `#ROContactMade`
- "Good service thank you" (alerter rating)

**Dynamic form (responder):**
- Incident Description: "Unknown Vehicle entering the place without plate license"

**Special procedures:** none

**Ground truth tags:**
- `#FeelingUnsafe` — driven by: comment "suspicious vehicle", form "unknown vehicle without plate"
- `#ROContactMade` — driven by: comment "contact made with client"
- `#ArrivedAtLocation` — driven by: log type 6 arrived on scene

---

## What the model input should look like

Based on real data, here is the proposed assembled input text per panic:

```
[META] creation=occurrence | type=ride | false_alarm=false | services=Armed Response,Private Medical

[PROCEDURES] Private Ambulance IN-TRIP: If rider or driver creates panic in trip and ambulance required dispatch Rocket

[LOGS] Incident Created | Incident Acknowledged | Private Medical service assigned | 
Armed Response service assigned | Armed Response accepted by Benneth Sibiya from Benneth Security | 
Responder arrived on scene | Armed Response completed | Incident Manually Resolved

[COMMENTS] bolt driver involved in an accident while on a trip please dispatch private ambulance and bill to CASI passenger is injured | 
Nare Olvies Malema bolt driver got accident the passenger is injured | 
Just spoke to the client ambulance is on scene | 9 min eta | 
spoke to bolt driver confirmed Medical emergency arrived | Ride Vehicle Details Make Suzuki S-Presso | 
Driver called the police | Standing down | 
Bolt driver got accident the passenger is injured dispatched rocket called the police

[FORM] Incident Description: Accident | Responding Partner: Benneth Security | Officer: Benneth
```

**Important preprocessing rules:**
1. Exclude log types: 32 (location noise), 60 (observer escalation), null-type generic ack lines
2. Deduplicate repetitive location change logs entirely
3. Strip URLs from comments (dynamic form links, google maps links)
4. Strip dot-only comments (".", "..")
5. Collapse repeated "Significant Location Change" logs into a single summary or drop entirely
6. Use log `description` text directly — it's already human readable
7. Special procedures: use `name` + `description`, mark if `critical=true`

---

## Tag patterns observed

| Tag | Primary signals |
|---|---|
| `#CallNotAnswered` | Comment "not answering"/"cant get hold" + log type 25 "Calling Alerter" without type 26 follow-up |
| `#AccidentalPress` | Comment "accidental"/"accidental press" + is_false_alarm=true + log type 15 |
| `#ArrivedAtLocation` | Log type 6 "arrived on scene" — near-deterministic |
| `#ROContactMade` | Comment "contact made"/"spoke to"/"all in order" + log type 6 arrived |
| `#VehicleAccident` | Comment "accident"/"injured"/"passenger" + Private Medical service dispatched |
| `#FeelingUnsafe` | Comment "suspicious"/"unsafe" + alerter creation method + no false alarm |
| `#PanicAlarm` | Generic — real incident, no accidental press |

---

## Noise to strip from training input

| Source | What to strip |
|---|---|
| `panic_logs` type=32 | All "Significant Location Change" entries |
| `panic_logs` type=60 | All "Observer Incident Escalation Level" entries |
| `panic_logs` type=null | Generic "Admin from X acknowledged" entries |
| `panic_comments` | URLs (http/https links), dots-only comments |
| `dynamic_form_answers` | Null/empty answers, image fields |
