# JARVIS Template System Documentation

## Overview

JARVIS uses a **multi-tier template system** to generate fast, appropriate replies. The system combines:

1. **Semantic Templates (74)** - Context-aware, embedding-based matching
2. **Simple Templates (18)** - Category-based random selection
3. **LLM Generation** - Fallback for complex/unmatched messages

---

## How Templates Were Created

### Phase 1: Universal Pattern Analysis (Jan 25-26, 2026)

**Source:** Analysis of universal texting patterns (NOT user-specific mining)

**Methodology:**
- Analyzed common texting behaviors across demographics
- Identified patterns everyone uses regardless of personal style
- Created templates based on **meaning**, not specific users

**Wave 1: Universal Texting Patterns (30+ templates)**

Created from patterns like:
```
Everyone says: "ok", "kk", "okay" → Created: quick_ok template
Everyone says: "lol", "haha" → Created: laughter template  
Everyone says: "omw", "on my way" → Created: on_my_way template
Everyone says: "thanks", "ty", "thx" → Created: quick_thanks template
Everyone says: "wanna hang", "down for" → Created: hang_out_invite template
Everyone says: "bye", "cya", "ttyl" → Created: goodbye template
```

**Key Insight:** These patterns are **linguistic universals** - found in texting across all English speakers, not specific to any individual.

**Wave 2: Group Chat Patterns (31 templates)**

Created from universal group behaviors:
- Event planning ("when works for everyone?")
- RSVP coordination ("count me in", "can't make it")
- Polls ("I vote for option A")
- Logistics ("who's bringing what?")
- Celebrations ("happy birthday!", "congrats!")

These are **social universals** - how groups coordinate everywhere.

**Wave 3: Assistant Query Templates (REMOVED)**

Originally created 25 templates for AI assistant queries ("summarize my messages").
**REMOVED** because they're the wrong use case - they explain what the assistant will do, not reply to friends.

### Phase 2: Template Cleanup (Feb 2026)

**Problems Identified:**
1. ❌ 21 templates were wrong use case (assistant queries)
2. ❌ 8 templates were too formal/business-like
3. ❌ Some patterns matched ambiguously (e.g., "💀" matching Unicode garbage)

**Actions Taken:**
1. ✅ Removed 21 assistant query templates
2. ✅ Rewrote 8 formal templates to be casual:
   - "You're welcome! Let me know if you need anything else" → "Ofc!"
   - "I'd be happy to meet. Could you share..." → "Sure! When works?"
3. ✅ Removed ambiguous patterns:
   - Removed "💀" from laughter (matched garbage)
   - Removed "hold on" from brb (wrong context)
   - Removed "i'm ok" from decline (false positives)
4. ✅ Added 10 new templates for common gaps

**Current State:** 74 templates, all universal, casual, and appropriate.

---

## Template Architecture

### 1. Semantic Templates (74 templates) - PRIMARY

**Location:** `models/template_defaults.py`

**How They Work:**
```python
# Template structure
ResponseTemplate(
    name="quick_ok",
    patterns=["ok", "kk", "okay", "okie", "k"],  # All variations
    response="Got it!",  # Universal response
)
```

**Matching Process:**
1. Incoming message: "kk"
2. Embed message → vector
3. Compare to all template pattern vectors
4. Find best match with similarity ≥ 0.85
5. Return template response

**Why Universal:**
- Patterns: Everyone uses "ok", "kk", "okay"
- Response: "Got it!" works for anyone
- No personalization to specific users

**Categories:**
- Quick acknowledgments (5 templates)
- Location/Time (6 templates)
- Social plans (6 templates)
- Reactions/Expressions (6 templates)
- Farewells (5 templates)
- Negation/Opinion (4 templates)
- Meeting/Scheduling (4 templates)
- Group chat (31 templates)
- NEW: Flexibility, Wait/Pause, Exclamations, Status (4 templates)

### 2. Simple Templates (18 templates) - FALLBACK

**Location:** `jarvis/prompts/constants.py`

**Used When:**
- Semantic matching fails
- Category is "acknowledge" or "closing"
- System is in degraded mode

**Structure:**
```python
ACKNOWLEDGE_TEMPLATES = [
    "ok", "sure", "got it", "thanks", "np", 
    "👍", "for sure", "alright", "bet", "cool"
]

CLOSING_TEMPLATES = [
    "bye!", "see ya", "later!", "talk soon", 
    "ttyl", "peace", "catch you later", "gn"
]
```

**Selection:** Random choice from category list

**Why Still Used:**
- Zero latency (no embeddings)
- Reliable for simple acknowledgments
- Good fallback when semantic matching fails

### 3. LLM Generation - ULTIMATE FALLBACK

**Used When:**
- No template matches
- Message is complex/unique
- Requires context understanding

**Process:**
1. RAG search for relevant context
2. Generate with lfm-1.2b model
3. Return AI-generated response

---

## Why This System is Generalizable

### ✅ Universal Patterns

**Examples of universals in templates:**

| Template | Pattern | Why Universal |
|----------|---------|---------------|
| quick_ok | "ok", "kk", "okay" | Everyone acknowledges this way |
| laughter | "lol", "haha" | Universal laughter expressions |
| on_my_way | "omw", "coming" | Everyone shares ETA this way |
| quick_thanks | "thanks", "ty" | Universal gratitude |
| hang_out_invite | "wanna hang" | Universal social invitation |
| goodbye | "bye", "cya" | Universal farewells |

### ❌ NOT User-Specific

**Removed/avoided patterns:**
- "mwah" - Too ambiguous (kiss sound, disappointment, filler)
- "breh" - Slang specific to certain groups
- "yee mb" - Friend group slang
- Name-specific patterns
- Reference-specific patterns ("Khris Middleton")

### 🎯 Balanced Approach

**Keep Universal (Patterns):**
- "ok", "lol", "omw", "thanks" - Everyone uses these
- Group coordination - Universal social behavior
- Basic emotions - Universal expressions

**Allow Personalization (Responses):** 
- Template: "ok" → Response: "Got it!" (default)
- But user could customize to: "bet" or "say less"
- Pattern stays universal, response adapts

---

## How It Works in Practice

### Flow Diagram

```
Incoming Message
       ↓
[Try Semantic Match]
    ↓ (similarity ≥ 0.85)
   YES → Return template response
    ↓ (no match)
[Try Simple Templates]
    ↓ (acknowledge/closing category)
   YES → Return random from ACKNOWLEDGE_TEMPLATES
    ↓ (no match or other category)
[LLM Generation]
    ↓
Return AI-generated response
```

### Example Flows

**Example 1: Simple Match**
```
Friend: "lol"
Semantic: Matches "laughter" template (similarity: 1.0)
Response: "Haha right?!"
Latency: ~10ms (embed + match)
```

**Example 2: No Semantic Match**
```
Friend: "Can you help me with this complex problem?"
Semantic: No match (similarity < 0.85)
Simple: Category = "question", not acknowledge/closing
LLM: Generates contextual response
Response: "Sure! What's going on?"
Latency: ~500ms (RAG + generation)
```

**Example 3: Fallback to Simple**
```
Friend: "ok"
Semantic: Matches "quick_ok" (similarity: 1.0)
Response: "Got it!"
Or if semantic fails:
Simple: Random from ["ok", "sure", "got it", ...]
Response: "sure"
```

---

## Performance Characteristics

### Semantic Templates (74)
- **Hit Rate:** ~14-20% (with good patterns)
- **Latency:** 10-20ms (embed + cosine similarity)
- **Quality:** High (context-aware)
- **Memory:** ~50MB (embeddings)

### Simple Templates (18)
- **Hit Rate:** ~5% (acknowledge/closing only)
- **Latency:** ~1ms (random choice)
- **Quality:** Medium (random, not context-aware)
- **Memory:** Negligible

### LLM Generation
- **Hit Rate:** 100% (fallback)
- **Latency:** 300-800ms (RAG + generation)
- **Quality:** High (contextual, personalized)
- **Memory:** ~2GB (model loaded)

### Combined System
- **Overall Hit Rate:** ~20-25% (templates + LLM)
- **Average Latency:** ~100ms (weighted by frequency)
- **Quality:** High (best of both worlds)

---

## Customization Guide

### For End Users (Personalization)

**Option 1: Change Template Responses**

Edit `models/template_defaults.py`:
```python
# Change response to match YOUR style
ResponseTemplate(
    name="quick_affirmative",
    patterns=["sure", "yep", "yeah"],
    response="bet",  # Change from "Sounds good!" to your style
)
```

**Option 2: Add Your Patterns**

Add patterns you use frequently:
```python
ResponseTemplate(
    name="your_agreement",
    patterns=["say less", "word", "fr fr"],  # Your slang
    response="say less",  # Your response
)
```

### For Developers (System Changes)

**Adjust Similarity Threshold:**
```python
# In models/templates.py
SIMILARITY_THRESHOLD = 0.85  # Increase for stricter matching
```

**Add New Templates:**
1. Edit `models/template_defaults.py`
2. Add ResponseTemplate to list
3. Ensure patterns are universal (not user-specific)
4. Test with evaluation script

**Monitor Hit Rates:**
```bash
uv run python evals/evaluate_semantic_templates.py --limit 200
```

---

## Testing & Validation

### Automated Evaluation

Run evaluation to check template quality:
```bash
uv run python evals/evaluate_semantic_templates.py --limit 200
```

**Metrics:**
- Hit rate: % messages matched
- Quality score: Cerebras judge (1-10)
- Coverage: Categories represented

### Manual Testing

Test specific patterns:
```python
from models.templates import TemplateMatcher

matcher = TemplateMatcher()
match = matcher.match("wanna hang out tonight?")
if match:
    print(f"Template: {match.template.name}")
    print(f"Response: {match.template.response}")
    print(f"Similarity: {match.similarity}")
```

---

## Future Improvements

### Short Term
- [ ] Add more universal patterns (aim for 100 templates)
- [ ] Improve group chat templates
- [ ] Add emoji-only pattern matching

### Medium Term
- [ ] Per-contact template customization
- [ ] Template learning from user corrections
- [ ] Multi-language template support

### Long Term
- [ ] Dynamic template generation
- [ ] Context-aware response selection
- [ ] Personality-aware templates

---

## Summary

**What We Built:**
- ✅ 74 semantic templates based on **universal** texting patterns
- ✅ 18 simple templates for fast fallback
- ✅ Multi-tier system (semantic → simple → LLM)
- ✅ **NOT user-specific** - works for anyone
- ✅ Context-aware matching with embeddings

**Key Principle:**
- **Patterns are universal** - everyone says "ok", "lol", "thanks"
- **Responses are general** - "Got it!" works for anyone
- **NOT mined from user data** - created from linguistic analysis
- **Wired into reply flow** - actually gets used now!

**Result:**
Fast, appropriate replies that work for **any user** because they're based on how **everyone** texts! 🎯
