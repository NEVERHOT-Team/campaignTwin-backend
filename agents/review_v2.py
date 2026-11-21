"""Interpret a planner's campaign text and ask for the next unresolved input."""

import json
import math
import re
from datetime import date, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from agents import llm

Field = Literal["channel", "product_hint", "pack_hint", "product_choice_id", "audience_hint",
                "budget_thb", "start_date", "end_date", "objective", "offer",
                "regular_price_thb", "cogs_thb", "order_fee_thb", "fulfillment_thb",
                "voucher_fee_thb", "paid_media_thb", "commission_pct", "profit_target_thb",
                "min_unit_contribution_thb", "product_scope", "alternative_policy",
                "max_loss_probability"]


class ProposedAnswer(BaseModel):
    field: Field
    value_json: str
    quote: str


class Interpretation(BaseModel):
    items: list[ProposedAnswer]


class Interview(BaseModel):
    first_field: str
    question: str


# Use a commercial vocabulary projection before sending or caching planner text.
# Unknown tokens break a quote. Raw contact details, household rows and identifiers
# must never enter the interview prompt or surface as a supporting quote.
WORDS = set("""
a an the it i we our my want need like yes no use choose select please run rehearse simulate
do don't don’t example sample demo synthetic listing item product twelve roll rolls pack
toilet tissue paper
shopee lazada tiktok tiktok_shop shop line meta google offline amazon shopify tokopedia ebay
facebook instagram walmart etsy online website web store on in at for to from through until starting
ending exclusive inclusive week weeks day days monday tuesday wednesday thursday friday saturday
sunday today tomorrow next this month january february march april may june july august september
october november december jan feb mar apr jun jul aug sep oct nov dec offer deal price off
discount percent percentage voucher coupon baht thb budget spend cap total subsidy maximum
limit keep change make raise lower set with of and or only not instead all shopper shoppers
buyers buyer everyone anybody customers new members member eligible targeting target audience
profit profitability incremental contribution revenue sales new_buyers repeat_rate awareness leads
objective goal maximize measure how much about per no minimum min purchase order threshold
also another additional second single multiple products coffee rice together net earn earns
shipping spend cost user redemptions each seller selling regular unit fee fees service payment
cogs goods fulfillment commission media paid least require required minimum floor
targeting net before after pack costs target earns earn
risk loss chance probability max at most cannot can’t can't alter locked advertised must remain
fixed payday money funds allocated allocate have is are will be should could would applied
grand thousand hundred half one two three four five six seven eight nine ten twenty thirty
fifty a hundred thousand zero tues thurs till between plus rather than avoid focus promote
promotion campaign all_buyers incremental_profit synthetic-tissue deal-hunters pick its
""".split())
TOKEN = re.compile(r"don['’]t|can['’]t|demo-tissue-12-shopee|\d{4}-\d{2}-\d{2}|(?:\d+[,.])*\d+|"
                   r"[\w]+|[^\w\s]", re.I)
PRIVATE = re.compile(r"\b(?:household|customer|shopper|member|order|account)[ _-]*"
                     r"(?:id|number|no\.?)"
                     r"\s*[:=#-]?\s*[\w-]+|"
                     r"\S+@\S+|https?://\S+|\b(?:sk-|Bearer\s+)\S+|"
                     r"(?<!\w)\+?\d(?:[ ()-]*\d){8,}(?!\w)", re.I)
DEMO = "demo-tissue-12-shopee"
NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
MONEY = r"(?:THB|baht|฿)"


# The live ledger accepts THB. Detect explicit foreign units before sanitizing
# their names (an unrecognized suffix would otherwise leave a plausible THB amount).
FOREIGN_UNIT = (r"(?:USD|dollars?|EUR|euros?|GBP|pounds?|JPY|yen|CNY|RMB|SGD|"
                r"MYR|IDR|VND|AUD|CAD|INR|HKD|KRW|CHF|NZD|PHP|BRL|TWD)")
FOREIGN_AMOUNT = re.compile(
    rf"(?:[$€£¥₹₫₱₩]\s*{NUMBER}|\b{FOREIGN_UNIT}\s*{NUMBER}|{NUMBER}\s*{FOREIGN_UNIT}\b)",
    re.I,
)
SCALED_AMOUNT = re.compile(rf"{NUMBER}\s*(?:k|m|b|thousand|grand|million|billion)\b", re.I)


def safe_text(text):
    text = PRIVATE.sub("…", text)

    def token(match):
        value = match[0]
        if (value.lower() in WORDS or value.lower() == DEMO
                or value[0].isdigit() or value in "%฿.,?!-/"):
            return value
        return "…"

    return TOKEN.sub(token, text)


def status(reason=None):
    return {"status": "degraded" if reason else "completed", "reason": reason}


FREE_TEXT_FIELDS = {"product_hint", "pack_hint", "audience_hint", "product_choice_id", "channel"}


def safe_value(field, value):
    return safe_text(value) if field in FREE_TEXT_FIELDS and isinstance(value, str) else value


def known_fields(facts):
    return {key: safe_value(key, value) for key, value in facts.items()}


def affirmative(text):
    return bool(re.fullmatch(r"(?:yes(?: please)?|yep|yeah|correct|confirm|that's right)[.!]?",
                             text.strip(), re.I))


def order_questions(questions, facts, pending):
    if not questions:
        return questions, None
    result = llm.call_llm("fast", [
        {"role": "system", "content": "Interview a campaign planner. Select one unresolved "
         "field and ask one short, natural, specific question. Use only the supplied facts "
         "and allowed choices in the prompts. If confirmation is needed, ask the planner "
         "to check the suggestions. Never grant approval or imply a simulation ran."},
        {"role": "user", "content": json.dumps({
            "questions": questions, "known": known_fields(facts),
            "proposed_fields": [item["field"] for item in pending],
        })},
    ], Interview)
    if result is None:
        return questions, status("provider_unavailable")
    prompt = result.question.strip()
    if result.first_field not in {q["field"] for q in questions} or not 1 <= len(prompt) <= 300:
        return questions, status("invalid_response")
    ordered = sorted(questions, key=lambda q: q["field"] != result.first_field)
    # Keep the exact yes/no wording for server-defined confirmation actions.
    pending_fields = {item["field"] for item in pending}
    if result.first_field not in pending_fields | {"/facts/product_choice_id"}:
        ordered[0] = {**ordered[0], "prompt": prompt}
    return ordered, status()


def local_answers(text, today, focus):
    items = []

    def add(key, value, quote):
        if quote and "…" not in quote:
            item = {"field": "/facts/" + key, "value": value, "brief_quote": quote}
            if not any(c["field"] == item["field"] and c["value"] == value for c in items):
                items.append(item)

    fixed = re.search(
        r"\b(?:(?:keep|leave) (?:the |our )?(?:advertised )?(?:discount|voucher|offer) fixed"
        r"|(?:do not|don['’]t|cannot|can['’]t) (?:change|alter|reduce) "
        r"(?:the |our )?(?:advertised )?(?:discount|voucher|offer))\b", text, re.I,
    )
    if fixed and not re.search(r"\b(?:not|never|don['’]t)\s*$", text[:fixed.start()], re.I):
        add("alternative_policy", "fixed_offer", fixed[0])

    multiple = r"\b(?:also promote|another product|multiple products|second product)\b"
    for pattern, scope in (
        (multiple, "multiple_products"),
        (r"\b(?:one|single) (?:focal )?product\b", "single_product"),
    ):
        match = re.search(pattern, text, re.I)
        if match:
            add("product_scope", scope, match[0])
    pair = re.search(r"\b(?:run|promote)\s+(?:toilet )?(?:tissue|coffee|rice)\s+"
                     r"(?:and|plus)\s+(?:toilet )?(?:tissue|coffee|rice)\b", text, re.I)
    if pair:
        add("product_scope", "multiple_products", pair[0])
    excluded_channels = []
    for field, pattern, canonical in [
        ("product_hint", r"\btoilet[ -](?:tissue|paper)\b", "toilet tissue"),
        ("product_choice_id", r"demo-tissue-12-shopee|\b(?:synthetic|demo)\b", DEMO),
        ("pack_hint", r"\b12[ -]rolls?\b", "12 rolls"),
        ("channel", r"\b(?:shopee|lazada|offline|line|meta|google|amazon|shopify|ebay|"
                    r"tokopedia|facebook|instagram|walmart|etsy)\b", None),
        ("channel", r"\btiktok[ _-]?shop\b", "tiktok_shop"),
        ("audience_hint", r"\b(?:all (?:eligible )?(?:buyers|shoppers|customers)|everyone)\b",
         "all"),
        ("audience_hint", r"\bdeal[ -]hunters?\b", "deal hunters"),
        ("audience_hint", r"\b(?:members? only|only members?)\b", "members only"),
        ("audience_hint", r"\b(?:new buyers? only|only new buyers?)\b", "new buyers only"),
        ("objective", r"\b(?:revenue|awareness|leads)\b", None),
        ("objective", r"\brepeat rate\b", "repeat_rate"),
        ("objective", r"\b(?:incremental |contribution )?(?:profit|profitability)\b",
         "incremental_profit"),
    ]:
        for match in re.finditer(pattern, text, re.I):
            excluded = re.search(
                r"\b(?:not|no|don['’]t|avoid|instead of)"
                r"(?: (?:use|choose|select|the|on|a|synthetic))*\s*$",
                text[:match.start()], re.I,
            )
            if excluded:
                if field == "channel":
                    excluded_channels.append((canonical or match[0].lower(),
                                              text[excluded.start():match.end()]))
                continue
            add(field, canonical or match[0].lower(), match[0])

    if not any(item["field"] == "/facts/channel" for item in items):
        for channel, quote in excluded_channels:
            add("channel", "exclude " + channel, quote)

    # Bind amounts to their business basis before considering a bare percentage an offer.
    consumed = []
    money_fields = {
        "regular_price_thb": r"(?:regular|selling|unit) price",
        "cogs_thb": r"(?:cogs|unit cost|cost of goods)",
        "order_fee_thb": r"(?:seller|order|payment) fee",
        "fulfillment_thb": r"fulfillment(?: cost| fee)?",
        "voucher_fee_thb": r"voucher (?:service )?fee",
        "paid_media_thb": r"(?:paid media|media spend)",
        "profit_target_thb": r"(?:profit target|target profit|incremental profit (?:target|goal))",
        "min_unit_contribution_thb": r"(?:contribution floor|minimum unit contribution)",
    }
    for key, label in money_fields.items():
        expression = rf"\b{label}\s*(?:of|is|at least)?\s*{MONEY}?\s*({NUMBER})\s*{MONEY}?"
        for match in re.finditer(expression, text, re.I):
            if text[match.end():].lstrip().startswith("%"):
                continue
            amount = float(match[1].replace(",", ""))
            if math.isfinite(amount):
                add(key, amount, match[0].strip())
                consumed.append(match.span())
    target_pattern = (rf"(?:at least|earn|target(?: of)?)\s*{MONEY}?\s*({NUMBER})"
                      rf"\s*{MONEY}?\s+(?:incremental |net |contribution )?profit")
    for match in re.finditer(target_pattern, text, re.I):
        amount = float(match[1].replace(",", ""))
        if math.isfinite(amount):
            add("profit_target_thb", amount, match[0].strip())
    for expression in (rf"(?:commission|seller fee)\s*(?:of|is)?\s*({NUMBER})\s*%",
                       rf"({NUMBER})\s*%\s*(?:commission|seller fee)"):
        for match in re.finditer(expression, text, re.I):
            amount = float(match[1].replace(",", ""))
            if 0 <= amount <= 100:
                add("commission_pct", amount, match[0].strip())
                consumed.append(match.span())
    risk = r"(?:chance of loss|probability of loss|loss (?:risk|probability))"
    for expression in (
        rf"\b(?:maximum|max) {risk}\s*(?:is|of)?\s*({NUMBER})\s*(?:%|percent)",
        rf"\b{risk} (?:limit(?: is| of)?|(?:is )?at most)\s*({NUMBER})\s*(?:%|percent)",
        rf"\b(?:at most |maximum |max )({NUMBER})\s*(?:%|percent)\s+{risk}\b",
        rf"\b({NUMBER})\s*(?:%|percent)\s+{risk} limit\b",
    ):
        for match in re.finditer(expression, text, re.I):
            add("max_loss_probability", float(match[1].replace(",", "")) / 100, match[0].strip())
            consumed.append(match.span())
    for match in re.finditer(rf"({NUMBER})\s*(?:%|percent)\s*(off\b|discount\b)?", text, re.I):
        if any(first <= match.start() < end for first, end in consumed):
            continue
        amount = float(match[1].replace(",", ""))
        explicit_discount = match[2] or re.search(
            r"\b(?:discount|voucher|coupon)\s*(?:of|is|to)?\s*$", text[:match.start()], re.I
        )
        standalone_reply = focus == "offer" and text.strip().rstrip(".!") == match[0].strip()
        if (explicit_discount or standalone_reply) and 0 < amount <= 100:
            add("offer", {"kind": "percent_off", "unit": "pct", "value": amount}, match[0].strip())

    patterns = {
        "budget_thb": [rf"\b(?:budget|spend cap)\s*(?:of|is|at)?\s*{MONEY}?\s*({NUMBER})",
                       rf"({NUMBER})\s*{MONEY}\s+budget\b"],
        "offer": [rf"\b(?:voucher|coupon)\s*(?:of|for|is)?\s*{MONEY}?\s*({NUMBER})",
                  rf"({NUMBER})\s*{MONEY}\s+(?:voucher|coupon)\b"],
    }
    if focus in patterns and re.fullmatch(rf"\s*{MONEY}?\s*{NUMBER}\s*{MONEY}?\s*[.!]?\s*",
                                          text, re.I):
        patterns[focus].append(rf"({NUMBER})")
    for field, expressions in patterns.items():
        for expression in expressions:
            for match in re.finditer(expression, text, re.I):
                if re.match(r"\s*(?:%|percent\b)", text[match.end():], re.I):
                    continue
                amount = float(match[1].replace(",", ""))
                if not math.isfinite(amount):
                    continue
                if field == "budget_thb":
                    add(field, amount, match[0].strip())
                elif amount > 0:
                    add(field, {"kind": "voucher", "unit": "THB", "value": amount},
                        match[0].strip())
    for item in items:
        if item["field"] != "/facts/offer":
            continue
        restrictions = [
            (rf"\b(?:min(?:imum)? (?:spend|purchase)|threshold)\s*(?:of|is)?\s*{MONEY}?"
             rf"\s*({NUMBER})", "min_spend_thb", float),
            (r"\b(?:cap per user|redemptions per user)\s*(?:of|is)?\s*(\d+)",
             "cap_per_user", int),
        ]
        for pattern, key, cast in restrictions:
            term = re.search(pattern, text, re.I)
            if term:
                value = cast(term[1].replace(",", ""))
                if 0 <= value < float("inf") and (key == "min_spend_thb" or value >= 1):
                    item["value"][key] = value
        if re.search(r"\b(?:members? only|only members?)\b", text, re.I):
            item["value"]["eligibility"] = "members"
        elif re.search(r"\b(?:new buyers? only|only new buyers?)\b", text, re.I):
            item["value"]["eligibility"] = "new_only"
    next_week = re.search(r"\bnext week\b", text, re.I)
    if next_week:
        start = today + timedelta(days=7 - today.weekday())
        add("start_date", start.isoformat(), next_week[0])
        add("end_date", (start + timedelta(days=7)).isoformat(), next_week[0])
    dates = list(re.finditer(r"\b\d{4}-\d{2}-\d{2}\b", text))
    for i, match in enumerate(dates[:2]):
        try:
            date.fromisoformat(match[0])
        except ValueError:
            continue
        end = i == 1 or (len(dates) == 1 and (
            focus == "end_date" or re.search(r"\b(?:end|until|to)\s*$", text[:match.start()], re.I)
        ))
        value = match[0]
        quote = match[0]
        inclusive = re.match(r"\s*\(?inclusive\b", text[match.end():], re.I)
        if end and inclusive and date.fromisoformat(value) < date.max:
            value = (date.fromisoformat(value) + timedelta(days=1)).isoformat()
            quote += inclusive[0]
        add("end_date" if end else "start_date", value, quote)
    return items


def monetary_evidence(quote):
    amounts = {float(n.replace(",", "")) for n in re.findall(NUMBER, quote)}
    units = "zero one two three four five six seven eight nine ten".split()
    terms = {word: i for i, word in enumerate(units)}
    terms.update({"half": 0.5, "a": 1, "twenty": 20, "thirty": 30, "fifty": 50})
    subtotal = 0
    for token in re.findall(r"[a-z]+|" + NUMBER, quote.lower()):
        if token in terms:
            subtotal += terms[token]
        elif token in {"hundred", "thousand", "grand"}:
            subtotal = (subtotal or 1) * (100 if token == "hundred" else 1000)
        elif token[0].isdigit():
            subtotal = float(token.replace(",", ""))
        elif token != "and":
            subtotal = 0
        if token in terms or token in {"hundred", "thousand", "grand"} or token[0].isdigit():
            amounts.add(subtotal)
    return amounts


def extract(message, review=None):
    from api.v2.contract import V2Error, validator

    commercial_text = PRIVATE.sub("…", message)
    if FOREIGN_AMOUNT.search(commercial_text):
        raise V2Error(422, "invalid_request", "This rehearsal uses THB. Enter amounts in THB.")
    if SCALED_AMOUNT.search(commercial_text):
        raise V2Error(422, "invalid_request", "Enter the full amount in THB, for example 1500.")
    review = review or {}
    today = datetime.now(ZoneInfo("Asia/Bangkok")).date()
    fields = [q["field"].split("/")[-1] for q in review.get("questions", [])]
    safe = safe_text(message)
    fallback = local_answers(safe, today, fields[0] if fields else None)
    questions = review.get("questions", [])
    current_prompt = questions[0]["prompt"] if questions else ""
    context = {
        "current_question": safe_text(current_prompt),
        "today_in_Bangkok": today.isoformat(), "current_question_fields": fields,
        "known": known_fields({**review.get("facts", {}), **review.get("commercial_inputs", {})}),
        "pending": [{"field": c["field"], "value": safe_value(
            c["field"].split("/")[-1], c["value"]
        )} for c in review.get("candidates", [])],
        "message": safe,
    }
    result = llm.call_llm("fast", [
        {"role": "system", "content": "Interpret a seller's campaign message in context. "
         "Return fields requested or corrected in THIS message only. Each needs a short exact "
         "verbatim quote from message with no ellipsis. Unknowns stay unresolved. Encode each "
         'normalized value as serialized JSON in value_json (e.g. "1000" for a number and '
         '\'"shopee"\' for a string). Interpret shorthand answers against the current question. '
         "Keep the requested sales channel even when unsupported (including Shopify or Amazon). "
         "Use lowercase and tiktok_shop for TikTok Shop. "
         "objective: incremental_profit/revenue/new_buyers/repeat_rate/awareness/leads. "
         "A synthetic/demo product selection maps to demo-tissue-12-shopee; never select it from "
         "generic product text alone. audience_hint 'all' means all shoppers. budget_thb is a "
         "number. offer is an object: kind percent_off or voucher, value, unit pct or THB. "
         "Preserve stated limits and eligibility. Dates are ISO local dates; end_date is "
         "exclusive. 'next week' is Monday through the following Monday exclusive. "
         "Do not invent an amount, omit a material restriction, assume demo consent or infer "
         "authorization to launch. An unsupported intent may remain a planner hint."},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ], Interpretation)
    if result is None:
        return fallback, status("provider_unavailable")
    items = []
    for proposal in result.items:
        quote = proposal.quote.strip()
        if not quote or "…" in quote or quote not in safe or quote not in message:
            return fallback, status("invalid_response")
        try:
            value = json.loads(proposal.value_json)
            json.dumps(value, allow_nan=False)
        except (ValueError, TypeError):
            return fallback, status("invalid_response")
        if not validator("PlannerFields").is_valid({proposal.field: value}):
            return fallback, status("invalid_response")
        if proposal.field == "offer" or proposal.field.endswith(("_thb", "_pct")):
            amount = value.get("value") if isinstance(value, dict) else value
            if amount is not None and amount not in monetary_evidence(quote):
                return fallback, status("invalid_response")
        if proposal.field in {"alternative_policy", "max_loss_probability"}:
            if not any(c["field"] == "/facts/" + proposal.field and c["value"] == value
                       for c in fallback):
                return fallback, status("invalid_response")
        if proposal.field == "offer":
            # A loss ceiling or commission percentage can appear in the very same brief.
            # Its numeric match alone cannot substantiate a discount of the same depth.
            if value.get("unit") == "pct" and any(
                c["field"] in {"/facts/max_loss_probability", "/facts/commission_pct"}
                for c in fallback
            ) and not any(
                c["field"] == "/facts/offer" and c["value"].get("unit") == "pct"
                and c["value"].get("value") == value.get("value") for c in fallback
            ):
                return fallback, status("invalid_response")
            for parsed in fallback:
                if parsed["field"] != "/facts/offer":
                    continue
                explicit = parsed["value"]
                if any(key in explicit and value.get(key) != explicit[key]
                       for key in ("eligibility", "min_spend_thb", "cap_per_user")):
                    return fallback, status("invalid_response")
        if proposal.field in FREE_TEXT_FIELDS and (
            "…" in safe_text(value)
        ):
            return fallback, status("invalid_response")
        if proposal.field == "product_choice_id" and (
            value != DEMO or not any(c["field"] == "/facts/product_choice_id" for c in fallback)
        ):
            return fallback, status("invalid_response")
        item = {"field": "/facts/" + proposal.field, "value": value, "brief_quote": quote}
        if not any(c["field"] == item["field"] and c["value"] == value for c in items):
            items.append(item)
    # Explicit platform choices and exclusions must survive interpretation; the resolver
    # can then explain compatibility and the planner can select among competing channels.
    retained = [c for c in fallback if c["field"].endswith(("_thb", "_pct"))
                or c["field"] in {"/facts/channel", "/facts/product_scope",
                                  "/facts/alternative_policy", "/facts/max_loss_probability"}
                or (c["field"] == "/facts/audience_hint" and
                    c["value"] in {"members only", "new buyers only", "deal hunters"})
                or (c["field"] == "/facts/offer" and
                    {"eligibility", "min_spend_thb", "cap_per_user"}.intersection(c["value"]))]
    if retained:
        retained_fields = {c["field"] for c in retained}
        items = [c for c in items if c["field"] not in retained_fields] + retained
    return items, status()
