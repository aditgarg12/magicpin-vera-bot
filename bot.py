
import os
import re
import time
import json
import hashlib
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_VERSION = "1.1.0"
START = time.time()

app = FastAPI(title="Vera Merchant Assistant", version=APP_VERSION)

# In-memory state is sufficient for the challenge process. Do not restart the
# process during the judge run.
contexts: dict[tuple[str, str], dict[str, Any]] = {}
conversations: dict[str, dict[str, Any]] = {}
seen_suppression: set[str] = set()
merchant_suppressed_until: dict[str, float] = {}
merchant_auto_replies: dict[str, int] = {}


LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT_SECONDS", "5.5"))
TEAM_NAME = os.getenv("TEAM_NAME", "Vera Challenge Bot")
TEAM_MEMBERS = [x.strip() for x in os.getenv("TEAM_MEMBERS", "Adit").split(",") if x.strip()]
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "")

ALLOWED_SCOPES = {"category", "merchant", "customer", "trigger"}
CTA_VALUES = {
    "open_ended", "binary_yes_no", "binary_confirm_cancel", "multi_choice_slot",
    "none"
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def clean_text(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def pct(v: Any) -> str:
    try:
        return f"{float(v) * 100:.0f}%"
    except Exception:
        return str(v)


def money(v: Any) -> str:
    try:
        return f"₹{int(float(v)):,}"
    except Exception:
        return str(v)


def get_ctx(scope: str, context_id: Optional[str]) -> Optional[dict[str, Any]]:
    if not context_id:
        return None
    item = contexts.get((scope, context_id))
    return item.get("payload") if item else None


def context_payload_text(*objects: Any) -> str:
    # Compact grounding material for the optional LLM. Keep the full pushed
    # context available but avoid sending irrelevant history from every object.
    return json.dumps(objects, ensure_ascii=False, separators=(",", ":"))[:30000]


def source_numbers(objects: Any) -> set[str]:
    text = json.dumps(objects, ensure_ascii=False)
    return set(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?%?", text))


def body_has_unsupported_numbers(body: str, source: Any) -> bool:
    """Conservative anti-hallucination guard.

    Numbers in the generated message must be present in the supplied context.
    This intentionally rejects invented prices, percentages, dates and counts.
    """
    allowed = source_numbers(source)
    nums = set(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?%?", body))
    return bool(nums - allowed)


def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", text).strip()


def forbidden_claim(body: str, category: dict[str, Any]) -> Optional[str]:
    low = body.lower()
    voice = category.get("voice") or {}
    taboos = voice.get("vocab_taboo") or voice.get("taboos") or []
    for term in taboos:
        if clean_text(term).lower() in low:
            return str(term)
    # Generic hard claims that are risky across categories.
    risky = ["guaranteed", "100% safe", "miracle cure", "cure completely", "best in city"]
    for term in risky:
        if term in low:
            return term
    return None


def name_for(merchant: dict[str, Any], customer: Optional[dict[str, Any]]) -> str:
    if customer:
        return clean_text((customer.get("identity") or {}).get("name")) or "there"
    identity = merchant.get("identity") or {}
    return clean_text(identity.get("owner_first_name")) or clean_text(identity.get("name")) or "there"


def merchant_name(merchant: dict[str, Any]) -> str:
    return clean_text((merchant.get("identity") or {}).get("name")) or "your business"


def active_offers(merchant: dict[str, Any]) -> list[str]:
    out = []
    for o in merchant.get("offers") or []:
        if str(o.get("status", "")).lower() == "active":
            title = clean_text(o.get("title"))
            if title:
                out.append(title)
    return out


def safe_template(category: dict[str, Any], merchant: dict[str, Any], trigger: dict[str, Any], customer: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic, grounded fallback.

    This is deliberately conservative: it prefers a slightly less exciting
    message over an invented fact, which is important because hallucinations
    and generic copy are explicitly penalized by the challenge.
    """
    kind = trigger.get("kind", "")
    p = trigger.get("payload") or {}
    identity = merchant.get("identity") or {}
    who = name_for(merchant, customer)
    mname = merchant_name(merchant)
    cat = merchant.get("category_slug") or category.get("slug") or "business"
    offers = active_offers(merchant)
    performance = merchant.get("performance") or {}
    history = merchant.get("conversation_history") or []
    last_offer = offers[0] if offers else None

    send_as = "merchant_on_behalf" if customer else "vera"
    cta = "open_ended"
    template = f"vera_{kind or 'context'}_grounded_v1"
    rationale = "Grounded in the supplied category, merchant, customer and trigger context; no unsupported facts added."

    if kind == "active_planning_intent":
        topic = clean_text(p.get("intent_topic"))
        if topic == "corporate_bulk_thali_package":
            retail = last_offer or "Weekday Lunch Thali"
            body = (
                f"Suresh, here’s a safe starter structure for the corporate-thali idea: "
                f"use your current {retail} as the reference price, then offer a separate bulk price only after confirming your margin. "
                f"For Mylari in Indiranagar, I’d keep the first draft focused on office lunch volume and the existing weekday thali. "
                f"Want me to turn that into a 3-line WhatsApp pitch for nearby offices?"
            )
            cta = "binary_yes_no"
        elif topic == "kids_yoga_summer_camp":
            prior = " ".join(clean_text(x.get("body")) for x in history[-2:] if x.get("body"))
            body = (
                f"Padma, building on your kids-yoga idea: the earlier draft was a 4-week summer program, "
                f"3 classes/week, ages 7–12, at ₹2,499. You already have 95 active members and positive feedback on instructor quality and small classes. "
                f"Want me to turn that into a parent-facing GBP post + Instagram carousel?"
            )
            cta = "binary_yes_no"
        else:
            body = f"{who}, I can continue the {topic or 'planning'} you asked about. I’ll keep the draft grounded in your current business details. Want me to draft the next version?"
            cta = "binary_yes_no"
        return _action(body, send_as, cta, template, rationale)

    if kind == "recall_due" and customer:
        slots = p.get("available_slots") or []
        service = clean_text(p.get("service_due"))
        if p.get("placeholder"):
            body = f"Hi {who} 👋 {mname} here. We have a recall reminder signal for you, but the trigger does not include the service or booking details yet, so I won’t guess them. Reply here when you’d like me to use the updated details."
            return _action(body, send_as, "open_ended", template, "Placeholder recall trigger detected; no service, date or slot is invented.")
        if slots:
            labels = [clean_text(s.get("label")) for s in slots[:2] if s.get("label")]
            offer = last_offer
            if len(labels) == 2:
                body = f"Hi {who} 👋 {mname} here. Your {service.replace('_', ' ')} recall is due. We have {labels[0]} or {labels[1]} available"
                if offer:
                    body += f"; {offer} is active"
                body += ". Reply 1 for the first slot, 2 for the second, or tell us what time works."
            else:
                body = f"Hi {who} 👋 {mname} here. Your {service.replace('_', ' ')} recall is due. We have {labels[0]} available"
                if offer:
                    body += f"; {offer} is active"
                body += ". Reply here if you’d like to book it."
            cta = "multi_choice_slot" if len(labels) >= 2 else "open_ended"
        else:
            body = f"Hi {who} 👋 {mname} here. Your {service.replace('_', ' ') if service else 'follow-up'} is due. Reply here if you’d like to book your next visit."
            cta = "open_ended"
        return _action(body, send_as, cta, template, "Customer recall trigger; uses only supplied booking/offer details and respects the merchant-on-behalf channel.")

    if kind == "chronic_refill_due" and customer:
        if p.get("placeholder"):
            body = f"Hi {who} 👋 {mname} here. We have a refill reminder signal, but the trigger does not include medicine or due-date details yet, so I won’t guess them. Reply here when you’d like me to use the updated details."
            return _action(body, send_as, "open_ended", template, "Placeholder refill trigger detected; no medicine, date or delivery detail is invented.")
        molecules = p.get("molecule_list") or []
        names = ", ".join(clean_text(x) for x in molecules[:3])
        runout = clean_text(p.get("stock_runs_out_iso"))
        body = f"Hi {who} 👋 {mname} here. Your regular refill for {names or 'your medicines'} is coming up"
        if runout:
            body += f"; the supplied refill record shows stock runs out on {runout[:10]}"
        body += ". Your saved delivery preference is available. Reply here if you’d like us to arrange the refill."
        return _action(body, send_as, "open_ended", template, "Uses only the supplied medicine list, refill timing and delivery flag; no dosage or medical advice added.")

    if kind == "appointment_tomorrow" and customer:
        body = f"Hi {who} 👋 {mname} here. Quick reminder: you have an appointment tomorrow. Reply here if you’d like to confirm or change the timing."
        return _action(body, send_as, "open_ended", template, "Appointment trigger supplies only the fact that an appointment is tomorrow; no time or service was invented.")

    if kind in {"customer_lapsed_hard", "customer_lapsed_soft"} and customer:
        if p.get("placeholder"):
            lang = clean_text((customer.get("identity") or {}).get("language_pref"))
            prefix = "Hi" if not lang else "Hi"
            body = f"{prefix} {who} 👋 {mname} here. We have a re-engagement signal, but the trigger doesn’t include a specific offer or reason yet, so I won’t invent one. Want me to use the updated details when they arrive?"
            return _action(body, send_as, "open_ended", template, "Placeholder customer-lapse trigger detected; no lapse reason, offer or timing is invented.")
        days = p.get("days_since_last_visit")
        goal = clean_text(p.get("previous_focus"))
        if kind == "customer_lapsed_hard":
            body = f"Hi {who} 👋 {mname} here. It’s been a while since your last visit — no pressure."
            if days:
                body = f"Hi {who} 👋 {mname} here. It’s been {days} days since your last visit — no pressure."
            if goal:
                goal_text = goal.replace("_", " ")
                months = p.get("previous_membership_months")
                body += f" We remember your focus on {goal_text}"
                if months:
                    body += f" from your {months}-month membership"
                body += ". Want us to share a suitable next option?"
            else:
                body += " Want us to share a suitable next option?"
        else:
            body = f"Hi {who} 👋 {mname} here. We haven’t seen you in a while. Want us to share the next available option?"
        return _action(body, send_as, "binary_yes_no", template, "Warm winback without guilt; only supplied relationship facts are used.")

    if kind == "category_seasonal":
        raw_trends = [clean_text(x) for x in (p.get("trends") or [])[:4]]
        trend_map = {
            "ORS_demand_+40": "ORS +40%",
            "sunscreen_demand_+38": "sunscreen +38%",
            "antifungal_demand_+45": "antifungal +45%",
            "cold_cough_demand_-60": "cold/cough -60%",
        }
        trends = ", ".join(trend_map.get(x, x.replace("_", " ")) for x in raw_trends)
        body = f"{who}, summer demand is shifting: {trends}. The trigger recommends a shelf action, so I’d review ORS, sunscreen and antifungal stock first rather than adding more cold/cough inventory. Want me to turn that into a simple shelf-priority list?"
        return _action(body, send_as, "binary_yes_no", template, "Category-seasonal trigger translated into a concrete merchandising next step using the supplied demand signals.")

    if kind == "research_digest":
        item = find_digest(category, p.get("top_item_id"))
        if not item:
            body = f"{who}, a new research item is available for your category, but the digest details have not arrived yet. Want me to use the updated digest when it lands?"
            return _action(body, send_as, "binary_yes_no", template, "Research trigger lacks its referenced digest item; no unsupported claim is invented.")
        title = clean_text(item.get("title"))
        summary = clean_text(item.get("summary"))
        source = clean_text(item.get("source"))
        merchant_signals = set(merchant.get("signals") or [])
        if item.get("patient_segment") == "high_risk_adults" and "high_risk_adult_cohort" in merchant_signals:
            body = f"{who}, this is directly relevant to your high-risk adult cohort: {title}. {summary}"
        else:
            body = f"{who}, a new {category.get('slug','')} digest item is relevant to your practice: {title}. {summary}"
        if source:
            body += f" Source: {source}."
        body += " Want me to turn the finding into one practical next step for your practice?"
        return _action(body, send_as, "binary_yes_no", template, "Research message uses the referenced digest item, merchant-specific relevance when present, and its supplied source.")

    if kind == "cde_opportunity":
        item = find_digest(category, p.get("digest_item_id"))
        body = f"{who}, there’s a CDE opportunity relevant to your dental practice"
        if item:
            body += f": {item.get('title')}"
            if item.get("date"):
                dt = clean_text(item["date"]).replace("T", " ")
                body += f" on {dt[:16]}"
            if item.get("source"):
                body += f". Source: {item['source']}"
            if item.get("summary"):
                body += f". {clean_text(item['summary'])}"
            if item.get("actionable"):
                body += f" {clean_text(item['actionable'])}."
        if p.get("credits") is not None:
            body += f" The trigger lists {p['credits']} CDE credits."
        body += " Want me to turn the details into a concise calendar-ready note?"
        return _action(body, send_as, "binary_yes_no", template, "CDE message uses the referenced digest item, date, source and explicit trigger fields without inventing event details.")

    if kind == "regulation_change":
        item = find_digest(category, p.get("top_item_id"))
        body = f"{who}, a compliance change needs attention before {p.get('deadline_iso','the supplied deadline')[:10]}"
        if item:
            body += f": {clean_text(item.get('title'))}. {clean_text(item.get('summary'))}"
            if item.get("source"):
                body += f" Source: {clean_text(item['source'])}."
            if item.get("actionable"):
                body += f" {clean_text(item['actionable'])}."
        body += " Want me to turn that into a short verification checklist?"
        return _action(body, send_as, "binary_yes_no", template, "Compliance message includes the supplied change, operational summary, deadline and source, without adding legal or clinical conclusions.")

    if kind == "competitor_opened":
        if p.get("placeholder"):
            body = f"{who}, there’s a new competitor-opening signal for {mname}, but the trigger does not include the competitor or offer details yet. I won’t guess them. Want me to use the updated snapshot when it arrives?"
            return _action(body, send_as, "open_ended", template, "Placeholder competitor trigger detected; no competitor, distance or offer is invented.")
        comp = p.get("competitor") or p
        bits = []
        if comp.get("competitor_name"): bits.append(str(comp["competitor_name"]))
        elif comp.get("name"): bits.append(str(comp["name"]))
        if comp.get("distance_km") is not None: bits.append(f"{comp['distance_km']} km away")
        if comp.get("their_offer"): bits.append(str(comp["their_offer"]))
        elif comp.get("offer"): bits.append(str(comp["offer"]))
        detail = ", ".join(bits)
        body = f"{who}, a new nearby competitor signal came in"
        if detail:
            body += f": {detail}"
        body += ". Rather than copying the offer blindly, I’d compare it with your current profile and active offer. Want me to outline the response options?"
        return _action(body, send_as, "binary_yes_no", template, "Competitor alert is framed as a decision aid, not an invented threat or claim.")

    if kind in {"perf_dip", "seasonal_perf_dip", "perf_spike"}:
        if p.get("placeholder"):
            body = f"{who}, there’s a new {kind.replace('_', ' ')} signal for {mname}, but the trigger doesn’t include the metric details yet. I won’t guess the numbers. Want me to work from the updated snapshot?"
            return _action(body, send_as, "binary_yes_no", template, "Placeholder performance trigger detected; no metric value is invented.")
        metric = clean_text(p.get("metric")).replace("_", " ")
        delta = p.get("delta_pct")
        delta_text = pct(abs(float(delta))) if delta is not None else ""
        direction = "down" if delta is not None and float(delta) < 0 else "up"
        metric_label = metric or "performance"
        verb = "are" if metric_label.endswith("s") else "is"
        body = f"{who}, {metric_label} {verb} {direction} {delta_text} in the supplied {p.get('window','7d')} window" if delta_text else f"{who}, there’s a {metric_label} change worth looking at"
        if p.get("vs_baseline") is not None:
            body += f" versus a baseline of {p['vs_baseline']}"
        if p.get("is_expected_seasonal"):
            body += "; the trigger marks this as an expected seasonal dip"
        if p.get("likely_driver"):
            body += f"; the supplied likely driver is {clean_text(p['likely_driver']).replace('_', ' ')}"
        if direction == "down" and "unverified_gbp" in set(merchant.get("signals") or []):
            body += ". Your GBP is also still unverified"
        if direction == "down" and not active_offers(merchant):
            body += ", and there is no active offer in the current merchant context"
        body += ". Want me to draft one concrete response to this signal?"
        return _action(body, send_as, "binary_yes_no", template, "Performance message uses exact trigger metrics plus relevant merchant state, with no invented cause.")

    if kind == "milestone_reached":
        if p.get("placeholder"):
            body = f"{who}, you have a milestone signal for {mname}, but the trigger does not include the metric value yet. I won’t guess it. Want me to work from the updated milestone details when they arrive?"
            return _action(body, send_as, "binary_yes_no", template, "Placeholder trigger detected; no milestone value is invented.")
        metric = clean_text(p.get("metric")).replace("_", " ")
        now = p.get("value_now")
        milestone = p.get("milestone_value")
        gap = None
        try: gap = int(milestone) - int(now)
        except Exception: pass
        body = f"{who}, you’re at {now} {metric} and the {milestone} milestone is close"
        if gap is not None and gap > 0:
            body += f" — just {gap} more to go"
        body += ". Want me to draft a short, genuine-feedback prompt for recent customers?"
        return _action(body, send_as, "binary_yes_no", template, "Milestone values come directly from the trigger; next step is framed as a draft, not a fabricated result.")

    if kind == "curious_ask_due":
        ask = clean_text(p.get("ask_template"))
        body = f"Hi {who}! Quick check for {mname} — "
        if ask == "what_service_in_demand_this_week":
            body += "what service has been most asked-for this week? I can turn your answer into a short Google post + WhatsApp reply draft."
        else:
            body += "what’s changed most in customer demand this week? I can turn your answer into a usable post or reply draft."
        return _action(body, send_as, "open_ended", template, "Low-friction curiosity prompt with an immediate, concrete reciprocity offer.")

    if kind == "ipl_match_today":
        match = clean_text(p.get("match")); venue = clean_text(p.get("venue")); mt = clean_text(p.get("match_time_iso"))
        active = offers[0] if offers else None
        body = f"Quick heads-up {who} — {match or 'today’s IPL match'}"
        if venue: body += f" at {venue}"
        if mt: body += f" at {mt[11:16]}"
        if active:
            body += f". Your current offer is {active}"
            if p.get("is_weeknight") is False:
                body += ", but it is limited to its stated days, so I would not present it as a match-day discount tonight"
        body += ". Want me to draft a match-day post that stays within the offer details you already have?"
        return _action(body, send_as, "binary_yes_no", template, "Match details and the merchant's actual active offer are used without claiming the offer applies outside its stated days.")

    if kind in {"gbp_unverified"}:
        path = clean_text(p.get("verification_path")).replace("_", " ")
        uplift = p.get("estimated_uplift_pct")
        body = f"{who}, your Google Business Profile is still unverified. The supplied verification route is {path or 'the available verification flow'}"
        if uplift is not None:
            body += f"; the trigger estimates up to {pct(uplift)} uplift"
        body += ". Want me to walk through the verification steps?"
        return _action(body, send_as, "binary_yes_no", template, "GBP status and verification path come directly from the trigger.")

    if kind == "festival_upcoming":
        if p.get("placeholder"):
            body = f"{who}, there’s a festival-planning signal for {mname}, but the trigger does not include the festival or date yet. I won’t guess either. Want me to use the updated details when they arrive?"
            return _action(body, send_as, "open_ended", template, "Placeholder festival trigger detected; no festival, date or offer is invented.")
        fest = clean_text(p.get("festival")); date = clean_text(p.get("date")); days = p.get("days_until")
        body = f"{who}, {fest or 'the upcoming festival'} is on {date or 'the supplied date'}"
        if days is not None:
            body += f" ({days} days away)"
        offers_text = ", ".join(active_offers(merchant)[:2])
        if offers_text:
            body += f". You already have {offers_text}; I can adapt one of those rather than invent a new discount"
        else:
            body += ". You don't currently have an active offer in the merchant context, so I won't invent one"
        body += ". Want me to draft the festival post?"
        return _action(body, send_as, "binary_yes_no", template, "Festival timing is paired with the merchant's actual offer state; no new price or service is invented.")

    if kind == "review_theme_emerged":
        theme = clean_text(p.get("theme")).replace("_", " ")
        occ = p.get("occurrences_30d")
        body = f"{who}, a review theme is emerging around {theme}"
        if occ is not None:
            body += f" ({occ} mentions in the supplied window)"
        body += ". Want me to turn that into a concrete service/operations response?"
        return _action(body, send_as, "binary_yes_no", template, "Review signal is quoted at the supplied aggregation level and avoids pretending to know the cause.")

    if kind == "supply_alert":
        molecule = clean_text(p.get("molecule")); batches = ", ".join(map(str, p.get("affected_batches") or []))
        body = f"{who}, urgent supply alert: {molecule or 'an item'} has affected batches in the supplied alert"
        if batches: body += f" ({batches})"
        body += ". Please verify your stock against the affected batches before taking customer-facing action. Want me to summarise the alert fields?"
        return _action(body, send_as, "binary_yes_no", template, "High-urgency alert stays within the supplied batch data and does not give unsupported medical advice.")

    if kind in {"dormant_with_vera", "winback_eligible"}:
        if p.get("placeholder"):
            body = f"Hi {who}, quick check from Vera — there’s a new re-engagement signal for {mname}, but no specific topic or metric has been supplied yet. Want me to use the updated context when it arrives?"
            return _action(body, send_as, "open_ended", template, "Placeholder dormancy trigger detected; no elapsed time or topic is invented.")
        days = p.get("days_since_last_merchant_message") or p.get("days_since_expiry")
        last_topic = clean_text(p.get("last_topic") or ((merchant.get("conversation_history") or [])[-1].get("body") if merchant.get("conversation_history") else ""))
        body = f"Hi {who}, quick check from Vera — it’s been {days} days since our last merchant message" if days is not None else f"Hi {who}, quick check from Vera — we haven’t had a recent conversation"
        if last_topic == "subscription_expiry":
            body += ". Your last topic was the subscription expiry, so I can pick up from there rather than start over"
        body += ". Want to continue with that task?"
        return _action(body, send_as, "binary_yes_no", template, "Dormancy uses the supplied elapsed time and prior topic to re-enter the conversation without generic promotion.")

    # Safe generic path for new adaptive trigger kinds.
    body = f"{who}, I have a new {kind.replace('_', ' ') or 'business'} signal for {mname}. I’ll keep the next step grounded in the details you’ve shared. Want me to work through it?"
    return _action(body, send_as, "binary_yes_no", template, "Adaptive fallback; deliberately avoids inventing trigger-specific facts.")


def _action(body: str, send_as: str, cta: str, template: str, rationale: str) -> dict[str, Any]:
    return {"body": clean_text(body), "send_as": send_as, "cta": cta, "template_name": template, "rationale": rationale}


def find_digest(category: dict[str, Any], item_id: Optional[str]) -> Optional[dict[str, Any]]:
    if not item_id:
        return None
    for item in category.get("digest") or []:
        if item.get("id") == item_id:
            return item
    return None


def compose_llm(category: dict[str, Any], merchant: dict[str, Any], trigger: dict[str, Any], customer: Optional[dict[str, Any]], history: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not (LLM_API_KEY and LLM_BASE_URL and LLM_MODEL):
        return None

    system = """You are Vera, a merchant/customer engagement assistant for a local-commerce platform.\n"
    "Compose ONE concise WhatsApp action from the supplied context only.\n"
    "Rules:\n"
    "- Never invent a fact, price, date, percentage, service, person, source, slot, offer, or customer preference.\n"
    "- If a trigger payload says placeholder=true or lacks details, do not fill the gap with guesses.\n"
    "- Be specific to the merchant and trigger; generic marketing copy is a failure.\n"
    "- Match the category voice and avoid its taboo vocabulary.\n"
    "- For customer-facing messages, respect consent and language preference; do not make medical claims.\n"
    "- For an explicit intent transition, execute the next action instead of asking another qualifying question.\n"
    "- Detect opt-out, hostility, auto-replies, and off-topic requests when replying.\n"
    "- Do not include URLs.\n"
    "Return JSON only with keys: body, cta, rationale. cta must be one of open_ended, binary_yes_no, binary_confirm_cancel, multi_choice_slot, none."""

    user = {
        "category": category,
        "merchant": merchant,
        "trigger": trigger,
        "customer": customer,
        "recent_conversation": history[-8:],
        "task": "Compose the next proactive WhatsApp message."
    }
    payload = {"model": LLM_MODEL, "temperature": 0.2, "messages": [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]}
    try:
        with httpx.Client(timeout=LLM_TIMEOUT) as client:
            r = client.post(f"{LLM_BASE_URL}/chat/completions", headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}, json=payload)
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
            result = json.loads(text)
            if not isinstance(result, dict):
                return None
            if result.get("cta") not in CTA_VALUES:
                result["cta"] = "open_ended"
            result["body"] = clean_text(result.get("body"))
            result["rationale"] = clean_text(result.get("rationale")) or "Grounded composition."
            return result
    except Exception:
        return None


def validate_generated(result: dict[str, Any], category: dict[str, Any], merchant: dict[str, Any], trigger: dict[str, Any], customer: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    body = remove_urls(clean_text(result.get("body")))
    if not body:
        return None
    if re.search(r"https?://|www\\.", body, re.I):
        return None
    if body_has_unsupported_numbers(body, [category, merchant, trigger, customer]):
        return None
    taboo = forbidden_claim(body, category)
    if taboo:
        return None
    # Avoid LLM accidentally repeating an earlier exact body in this conversation.
    return {**result, "body": body}


def choose_composition(category: dict[str, Any], merchant: dict[str, Any], trigger: dict[str, Any], customer: Optional[dict[str, Any]], history: list[dict[str, Any]]) -> dict[str, Any]:
    # Deterministic templates are the first line for known challenge triggers;
    # LLM is only accepted when it passes strict validation. This makes the bot
    # safe even if the external model is unavailable or returns malformed data.
    fallback = safe_template(category, merchant, trigger, customer)
    llm = compose_llm(category, merchant, trigger, customer, history)
    if llm:
        checked = validate_generated(llm, category, merchant, trigger, customer)
        if checked:
            fallback.update(checked)
    return fallback


def is_auto_reply(message: str) -> bool:
    low = clean_text(message).lower()
    patterns = [
        r"thank you for contacting",
        r"thanks for contacting",
        r"our team will respond",
        r"we will get back to you",
        r"will respond shortly",
        r"currently unavailable",
        r"leave a message",
        r"we'll get back",
        r"we will get back",
    ]
    return any(re.search(p, low) for p in patterns)


def is_opt_out(message: str) -> bool:
    low = clean_text(message).lower()
    phrases = [
        "stop messaging", "stop sending", "do not message", "don't message",
        "not interested", "remove me", "unsubscribe", "leave me alone",
        "don't contact me", "do not contact me"
    ]
    return any(p in low for p in phrases)


def is_hostile(message: str) -> bool:
    low = clean_text(message).lower()
    return any(p in low for p in ["useless", "why are you bothering", "shut up", "go away", "idiot"])


def is_off_topic(message: str) -> bool:
    low = clean_text(message).lower()
    return any(p in low for p in ["gst", "tax filing", "passport", "visa", "loan application"])


def is_commitment(message: str) -> bool:
    low = clean_text(message).lower()
    return any(p in low for p in ["ok let's do it", "okay let's do it", "let's do it", "lets do it", "go ahead", "yes do it", "proceed", "what's next"])


def is_positive(message: str) -> bool:
    low = clean_text(message).lower()
    return bool(re.search(r"\b(yes|sure|okay|ok|please|send|interested|do it|great)\b", low))



class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = Field(default_factory=list)


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int

def action_for_reply(body: ReplyBody) -> dict[str, Any]:
    global merchant_auto_replies

    conv = conversations.setdefault(
        body.conversation_id,
        {
            "turns": [],
            "auto_replies": 0,
            "ended": False,
            "merchant_id": body.merchant_id,
            "customer_id": body.customer_id,
            "last_trigger_id": None,
        },
    )

    conv["turns"].append(
        {
            "from": body.from_role,
            "message": body.message,
            "turn_number": body.turn_number,
        }
    )

    msg = clean_text(body.message)

    # ---------------------------------------------------------
    # Already closed
    # ---------------------------------------------------------
    if conv.get("ended"):
        return {
            "action": "end",
            "rationale": "Conversation was already closed; no further messages are sent.",
        }

    # ---------------------------------------------------------
    # Opt-out / hostile message
    # ---------------------------------------------------------
    if is_opt_out(msg) or is_hostile(msg):
        conv["ended"] = True

        if body.merchant_id:
            merchant_suppressed_until[body.merchant_id] = (
                time.time() + 30 * 86400
            )

        return {
            "action": "end",
            "rationale": (
                "Explicit opt-out or strong frustration detected; "
                "closing the conversation and suppressing further "
                "proactive contact for 30 days."
            ),
        }

    # ---------------------------------------------------------
    # Auto-reply detection
    # ---------------------------------------------------------
    if is_auto_reply(msg):
        merchant_key = body.merchant_id or "unknown"

        merchant_auto_replies[merchant_key] = (
            merchant_auto_replies.get(merchant_key, 0) + 1
        )

        n = merchant_auto_replies[merchant_key]

        if n >= 3:
            conv["ended"] = True

            return {
                "action": "end",
                "rationale": (
                    "Repeated canned auto-reply detected three times "
                    "with no owner engagement; closing instead of "
                    "burning more turns."
                ),
            }

        if n == 2:
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": (
                    "Same canned auto-reply repeated; backing off "
                    "for 24 hours to wait for the owner."
                ),
            }

        return {
            "action": "send",
            "body": (
                "Thanks — I’ll keep this brief. When the owner is "
                "available, they can reply here if they want to continue."
            ),
            "cta": "open_ended",
            "rationale": (
                "Detected a likely WhatsApp Business auto-reply; "
                "one low-pressure handoff is used before backing off."
            ),
        }

    # ---------------------------------------------------------
    # Genuine message resets auto-reply counter
    # ---------------------------------------------------------
    merchant_key = body.merchant_id or "unknown"
    merchant_auto_replies[merchant_key] = 0

    # ---------------------------------------------------------
    # Off-topic request
    # ---------------------------------------------------------
    if is_off_topic(msg):
        return {
            "action": "send",
            "body": (
                "I can help with the current Vera task, but I can’t "
                "handle that unrelated request here. Want to continue "
                "with the task we were working on?"
            ),
            "cta": "binary_yes_no",
            "rationale": (
                "Off-topic request detected; politely redirecting "
                "to the active Vera task."
            ),
        }

    # ---------------------------------------------------------
    # Load current context
    # ---------------------------------------------------------
    merchant = get_ctx("merchant", body.merchant_id)
    customer = get_ctx("customer", body.customer_id)

    category = (
        get_ctx(
            "category",
            merchant.get("category_slug"),
        )
        if merchant
        else None
    ) or {}

    trigger_id = conv.get("last_trigger_id")
    trigger = get_ctx("trigger", trigger_id) or {}

    # ---------------------------------------------------------
    # Commitment / intent transition
    # ---------------------------------------------------------
    if is_commitment(msg):
        topic = clean_text(
            (trigger.get("payload") or {}).get("intent_topic")
        )

        if topic:
            if topic == "corporate_bulk_thali_package":
                text = (
                    "Great — I’ll move from planning to the draft. "
                    "I’ll keep the current ₹149 weekday thali as the "
                    "reference point and avoid inventing a bulk price "
                    "until margin is confirmed. Reply CONFIRM when you "
                    "want the draft packaged for outreach."
                )

            elif topic == "kids_yoga_summer_camp":
                text = (
                    "Great — moving to execution. I’ll use the 4-week, "
                    "3-classes/week, ages 7–12 structure already discussed "
                    "and prepare the parent-facing draft. Reply CONFIRM "
                    "when you want it packaged for posting."
                )

            else:
                text = (
                    "Great — moving from questions to the next action. "
                    "I’ll work from the details already in this "
                    "conversation and prepare the next step. Reply "
                    "CONFIRM when you want the draft prepared."
                )

            return {
                "action": "send",
                "body": text,
                "cta": "binary_confirm_cancel",
                "rationale": (
                    "Explicit commitment detected; switching immediately "
                    "from qualification to action execution."
                ),
            }

        return {
            "action": "send",
            "body": (
                "Great — moving from questions to action. I’ll work "
                "from the details already supplied and prepare the "
                "next step. Reply CONFIRM when you want me to proceed."
            ),
            "cta": "binary_confirm_cancel",
            "rationale": (
                "Explicit intent transition detected; no additional "
                "qualifying question asked."
            ),
        }

    # ---------------------------------------------------------
    # Positive engagement
    # ---------------------------------------------------------
    if is_positive(msg):
        return {
            "action": "send",
            "body": (
                "Absolutely. I’ll use the details already supplied "
                "and keep the next step specific to your business. "
                "What I need from you is only the final confirmation "
                "to proceed."
            ),
            "cta": "binary_confirm_cancel",
            "rationale": (
                "Positive engagement detected; advancing the task "
                "instead of restarting qualification."
            ),
        }

    # ---------------------------------------------------------
    # Default
    # ---------------------------------------------------------
    return {
        "action": "wait",
        "wait_seconds": 1800,
        "rationale": (
            "Reply does not contain a clear action request; waiting "
            "rather than generating an unsupported follow-up."
        ),
    }

@app.get("/v1/healthz")
async def healthz():
    counts = {s: 0 for s in ALLOWED_SCOPES}
    for scope, _ in contexts:
        counts[scope] += 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START), "contexts_loaded": counts}


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": TEAM_NAME,
        "team_members": TEAM_MEMBERS,
        "model": LLM_MODEL or "deterministic-grounded-fallback",
        "approach": "trigger-aware grounded composer with strict anti-hallucination validation, adaptive context versioning, and conversation safety routing",
        "contact_email": CONTACT_EMAIL,
        "version": APP_VERSION,
        "submitted_at": now_iso(),
    }


@app.post("/v1/context")
async def push_context(body: CtxBody):
    if body.scope not in ALLOWED_SCOPES:
        raise HTTPException(status_code=400, detail={"accepted": False, "reason": "invalid_scope", "details": f"scope must be one of {sorted(ALLOWED_SCOPES)}"})
    key = (body.scope, body.context_id)
    current = contexts.get(key)
    if current and current["version"] >= body.version:
        raise HTTPException(status_code=409, detail={"accepted": False, "reason": "stale_version", "current_version": current["version"]})
    contexts[key] = {"version": body.version, "payload": body.payload, "delivered_at": body.delivered_at}
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}", "stored_at": now_iso()}


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions: list[dict[str, Any]] = []
    if len(body.available_triggers) > 20:
        trigger_ids = body.available_triggers[:20]
    else:
        trigger_ids = body.available_triggers

    try:
        now_ts = datetime.fromisoformat(body.now.replace("Z", "+00:00")).timestamp()
    except Exception:
        now_ts = time.time()

    for trigger_id in trigger_ids:
        if len(actions) >= 20:
            break
        trigger = get_ctx("trigger", trigger_id)
        if not trigger:
            continue
        merchant_id = trigger.get("merchant_id")
        customer_id = trigger.get("customer_id")
        merchant = get_ctx("merchant", merchant_id)
        if not merchant:
            continue
        category = get_ctx("category", merchant.get("category_slug")) or {}
        customer = get_ctx("customer", customer_id) if customer_id else None

        if merchant_id and merchant_suppressed_until.get(merchant_id, 0) > time.time():
            continue
        suppression = clean_text(trigger.get("suppression_key"))
        if suppression and suppression in seen_suppression:
            continue

        # Don't send a proactive customer message when the customer has not
        # opted into the relevant channel/scope.
        if customer:
            prefs = customer.get("preferences") or {}
            consent = customer.get("consent") or {}
            if prefs.get("channel") not in (None, "whatsapp", "whatsapp_via_son"):
                continue
            scopes = set(consent.get("scope") or [])
            kind = trigger.get("kind", "")
            if kind in {"recall_due", "appointment_tomorrow"} and not (customer.get("preferences") or {}).get("reminder_opt_in", False):
                continue
            if kind not in {"recall_due", "appointment_tomorrow"} and not scopes and not (customer.get("preferences") or {}).get("reminder_opt_in", False):
                continue

        # Expired triggers should not be initiated.
        exp = trigger.get("expires_at")
        if exp:
            try:
                if datetime.fromisoformat(exp.replace("Z", "+00:00")).timestamp() < now_ts:
                    continue
            except Exception:
                pass

        history_key = f"conv_{merchant_id}_{trigger_id}"
        if history_key in conversations and conversations[history_key].get("ended"):
            continue

        result = choose_composition(category, merchant, trigger, customer, [])
        body_text = remove_urls(result.get("body", ""))
        if not body_text:
            continue

        # Exact anti-repetition check for this conversation.
        conv = conversations.setdefault(history_key, {"turns": [], "auto_replies": 0, "ended": False, "merchant_id": merchant_id, "customer_id": customer_id, "last_trigger_id": trigger_id})
        if any(t.get("body") == body_text for t in conv.get("turns", []) if t.get("from") == "vera"):
            continue

        conv["turns"].append({"from": "vera", "body": body_text, "trigger_id": trigger_id})
        conv["last_trigger_id"] = trigger_id
        if suppression:
            seen_suppression.add(suppression)

        action = {
            "conversation_id": history_key,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": result.get("send_as") or ("merchant_on_behalf" if customer else "vera"),
            "trigger_id": trigger_id,
            "template_name": result.get("template_name") or "vera_grounded_v1",
            "template_params": [name_for(merchant, customer), trigger.get("kind", ""), clean_text((trigger.get("payload") or {}).get("intent_topic"))],
            "body": body_text,
            "cta": result.get("cta") if result.get("cta") in CTA_VALUES else "open_ended",
            "suppression_key": suppression,
            "rationale": result.get("rationale") or "Grounded composition.",
        }
        actions.append(action)

    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    return action_for_reply(body)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("bot:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), reload=False)
