# สรุปงาน: Strategist, Rival, Shopper/SSR และการแก้ economics

ทำโดย VizKey01 (+ Claude) · 21 พ.ย. 2026 · อ่านคู่กับ `AGENTS.md` และ docstring ของแต่ละไฟล์

---

## 1. ทำอะไรไปแล้วบ้าง (อยู่ใน `dev` แล้ว)

| ลำดับ | งาน | ไฟล์ | สถานะ |
|---|---|---|---|
| 1 | **CampaignSpec v0.1** sync จาก `main` เข้า `dev` | `contracts/campaign_spec.py` | ✅ |
| 2 | **Strategist gen 0:** สร้าง baseline + แผน user + 6 archetypes แล้ว enforce ขอบเขต (ส่วนลด, mechanic, margin, งบ, channel, segment) และเช็ก diversity ≥ 2 knob | `agents/strategist.py` | ✅ test ผ่าน |
| 3 | **Rival:** S1–S3 อ่าน spec v0.1 (`spec_inputs()` ของ Gan) + `get_rival_scenarios(spec, strategy)` คำนวณแยกต่อแผน (baseline = none) | `data/rival.py` | ✅ |
| 4 | **Rival S4:** depth เป็น **% (0–100)** เหมือน S1–S3, แปลง library ที่เป็น fraction ให้เป็น %, เพดาน = 100 − `margin_floor_pct` ของคู่แข่งจาก `brands.csv` | `agents/rival_policy.py`, `mocks/context_pack.json` | ✅ |
| 5 | **Shopper engine:** ประชากรจาก personas, ตัดสินใจซื้อรายสัปดาห์ (need × elasticity × รอโปร), logit เลือกแบรนด์ เรา/คู่แข่ง/อื่น, ซื้อตุน → post-dip, ใช้ seed ร่วมกับ baseline | `engine/shopper.py` | ✅ |
| 6 | **SSR:** persona ตอบเป็นข้อความ → embed → เทียบ anchors → appeal [-1, 1] มี fallback และ `python -m agents.ssr --check` | `agents/ssr.py`, `agents/llm.py` (`embed()`) | ✅ |

ผล test บน dev: **144 passed, 2 skipped** (ใช้ `--basetemp=.pytest_tmp` บนเครื่อง Codex)

---

## 2. ปัญหาที่เจอตอนรัน arena

`python -m engine.arena` ได้ผลว่า **ทุกแผนขาดทุน −90k ถึง −194k และ p_loss = 1.0** สาเหตุมี 3 ชั้น:

1. **ค่า media กินกำไร:** evaluator หักค่า media = งบ × สัดส่วน media (เช่น 450k × 0.35 = 157.5k) แต่ยอดขายมาจากลูกค้าจำลองแค่ 2–5k คน และ media ไม่ได้ซื้อ reach อะไรเลย
2. **ส่วนลดลึกกว่าจุดคุ้มทุน:** Strategist คิดเพดานจาก margin 40% − fees 10% = ~33% โดย **ไม่ได้รวมค่าส่ง 25 บาทต่อ order** เพดานจริงอยู่ที่ประมาณ **25%**
3. **test_arena fail:** แผน `g0-deep` (33%) ขาดทุนทุกชิ้น พอคู่แข่ง undercut แย่งยอดไป ขาดทุนน้อยลง S3 เลยออกมาดีกว่า S1

กำไรต่อชิ้น (ราคา 315.55, ต้นทุน 189.33, fees 10%, ค่าส่ง ~22.7 บาทต่อชิ้น):

| ส่วนลด | 0% | 20% | 25% | 30% | 40% |
|---|---|---|---|---|---|
| กำไร/ชิ้น | +72 | +15 | ~+1 | −13 | −44 |

---

## 3. สิ่งที่แก้ใน branch `fix/economics-reach` (ยังไม่ได้ merge)

| แก้อะไร | ไฟล์ | ผล |
|---|---|---|
| เพดานส่วนลดคิดต้นทุนครบ: unit cost + fees + ค่าส่งต่อชิ้น (units ต่อ order ถ่วงน้ำหนักจาก personas) | `agents/strategist.py` | แผนที่ AI สร้างไม่ขาดทุนต่อชิ้น (เพดาน ~25%) |
| `budget_flex_pct = 0` ล็อกงบได้จริง (เดิมถูกเปลี่ยนเป็น 10%) | `agents/strategist.py` | |
| **Scale:** `ShopperParams.households_per_agent = 50` (label default) คูณ WeekStats ทุกตัว ไม่แตะ `personas.json` | `engine/shopper.py` | ยอดขายเทียบกับงบ 500k ได้ |
| **Media → reach:** reach = 1 − (1 − organic 0.3) × e^(−impressions / targeted) โดย impressions = งบ media ÷ `media_cost_per_reach_thb` | `engine/shopper.py` | media มีผลจริงและผลตอบแทนลดลงเมื่อใส่มากขึ้น |
| **Habit:** คนที่ซื้อแบรนด์อื่นมีโอกาสเปลี่ยนแบรนด์ประจำ = 0.2 × (1 − loyalty) | `engine/shopper.py` | undercut ของคู่แข่งยังเจ็บต่อหลังจบแคมเปญ |

ทุกการเปลี่ยนแปลงไม่ต้องแก้ evaluator หรือ arena (`offer_from_strategy` เพิ่มแค่ `reach=1.0` เป็นค่า optional)

---

## 4. เรื่อง demo brief: 25% หรือ 30%

- `test_arena` ใช้แผน **`g0-deep` จาก Strategist** ไม่ใช่แผน 40% ของ user หลังแก้ข้อ 3 แผนนี้จะลดลงมาที่ ~25% ซึ่งไม่ขาดทุนต่อชิ้นแล้ว test นี้น่าจะผ่านโดยไม่ต้องแก้ demo brief
- ถ้าจะเปลี่ยนแผนของ user ใน demo brief: **เลือก 25%** เพราะ 30% ยังขาดทุน ~13 บาทต่อชิ้น
- **แต่ 25% อาจยังไม่ทำให้ S1 เป็นบวก** เพราะยอดที่ขายให้คนที่ซื้ออยู่แล้วเสียกำไรชิ้นละ ~71 บาทเทียบกับ baseline ต้องรัน arena ดูจริง
- **story ทางเลือกที่ข้อมูลรองรับ:** แผน 40% ของ user ขาดทุน ~44 บาทต่อชิ้นตั้งแต่ชิ้นแรก ส่วน AI เจอแผน targeted 15–20% ที่ยังกำไรและรอดแม้คู่แข่ง undercut

---

## 5. ยังค้าง / ต้องตัดสินใจ

1. รัน test + arena บน `fix/economics-reach` แล้วค่อย merge เข้า dev
2. **หมวดสินค้า:** campaign library, mocks, replay ยังเป็นกาแฟ แต่ส่วนอื่นเป็นทิชชู่ (S4 แบบ LLM จะยังไม่มีวันรันจนกว่าจะแก้)
3. `RivalScenario` ต้องมี `strategy_id` / `label` (Big)
4. `mocks/report.json` และ `demo/replay.json` ยังเก็บ `discount_depth: 0.44` เป็น fraction ควรเป็น %
5. id ของแผนใน `demo/replay.json` (`deep-1111`) ไม่ตรงกับ Strategist (`g0-deep`)
6. Strategist: ต่อ LLM proposer และ gen 1 (reflect/mutate)
7. warning `red_team_failed ModuleNotFoundError` ใน arena ต้องดู traceback เต็ม
