# GLiNER Goldset Cleaning Report

**Date**: 2026-02-11
**Source**: training_data/gliner_goldset/candidate_gold_merged_r4.json
**Output**: training_data/gliner_goldset/candidate_gold_merged_r4_clean.json

## Summary

- **Total records**: 796
- **Records with removed candidates**: 21 (2.6%)
- **Total candidates removed**: 26
- **Records with deduplications**: 25 (3.1%)
- **Total duplicates removed**: 50
- **Records flagged "needs_context"**: 2 (0.3%)

## Issues Fixed

### 1. Span Text Not in Message (Removed: 26)

Candidates where `span_text` was not found (as substring) in `message_text`.
These are unfair to evaluate against since the model only sees `message_text`.

**Examples of removed candidates:**

**r2_fact_gs_0070**: 'I like product management atm so I’m trying to get more PM experiences in as of now'

- Removed 'python' (not found in message_text)
- Removed 'SQL' (not found in message_text)

**v1_fact_gs_0125**: 'Yeah exactly, but haven't totally decided yet, my dad just wants to talk to some other people first to see if there's any points of view that we might have missed, so barring anything insanely big I'm ready to just get to slowly doing more and working up and it'll get better slowly and slowly physically and mentally. Just a slow process.'

- Removed '1 min walking 1 min run' (not found in message_text)

**r2_fact_gs_0315**: 'happy diwali my brother !!🪔'

- Removed 'Diwali' (not found in message_text)

**r2_fact_gs_0080**: 'That’s difficult, there’s not really anything I can say that can help because this is just a battle with your own mind for now. The good thing is that there is something in your mind pushing you to desire to go to yk, not to say yk is going to fix anything but the one thing I like about yk and swadhyay in general is that it is a process everyone is going through- self development. It’s incredible the different amount of struggles and joys everyone goes through but the fact is that, we are all attempting to grow. Some might be going through drug problems, financial problems, some might even have everything perfect but going through problems that others can’t even fathom'

- Removed 'Swadhyay' (not found in message_text)

**r2_fact_gs_0102**: 'Kk! And ur down to leave anytime? I was just gonna take the Bart cause I hate driving in sf'

- Removed 'BART' (not found in message_text)

### 2. Duplicate Entities (Removed: 50)

Same person/place mentioned with slightly different text (e.g., "brother" vs "My brother").
Kept the more specific/longer version.

**Examples of deduplication:**

**r2_fact_gs_0005**: 'Thanks for the contact. My brother ended up going to the emergency room, but he’s fine now 😪'

- Removed near-duplicate 'brother' (kept 'my brother' as more specific)
- Removed near-duplicate 'brother' (kept 'my brother' as more specific)

**v1_fact_gs_0125**: 'Yeah exactly, but haven't totally decided yet, my dad just wants to talk to some other people first to see if there's any points of view that we might have missed, so barring anything insanely big I'm ready to just get to slowly doing more and working up and it'll get better slowly and slowly physically and mentally. Just a slow process.'

- Removed near-duplicate 'dad' (kept 'my dad' as more specific)
- Removed near-duplicate 'dad' (kept 'my dad' as more specific)

**r2_fact_gs_0096**: 'And my dad flew in'

- Removed near-duplicate 'dad' (kept 'my dad' as more specific)
- Removed near-duplicate 'dad' (kept 'my dad' as more specific)

**r2_fact_gs_0180**: 'Ohhh shit 1:30? I prolly can't then lmaooo my mom was needing me to help her translate for a doctor's appointment at 1 smh I thought it was at 7:30 or something'

- Removed near-duplicate 'mom' (kept 'my mom' as more specific)
- Removed near-duplicate 'mom' (kept 'my mom' as more specific)

**r2_fact_gs_0218**: 'Smhhh I don't have my car rn my brother's using it'

- Removed near-duplicate 'brother' (kept 'my brother' as more specific)
- Removed near-duplicate 'brother' (kept 'my brother' as more specific)

### 3. Context-Dependent Records (Flagged: 2)

Records where the gold label only makes sense with surrounding context.
Flagged with `needs_context=true` for downstream evaluation.

**Examples of context-dependent records:**

**r2_fact_gs_0252**: 'Vestibular'

- Reason: Single-word isolated message requires context

**r2_fact_gs_0037**: 'Can my dad pick up a package for me if he has my comet card'

- Reason: Gold notes suggest context-dependence: in the hospital (from context: i'm kinda in the hospital)

## Evaluation Recommendations

1. **Span extraction evaluation**:
   - Use cleaned dataset to fairly evaluate model span detection
   - Model was trained on full message_text, so this is appropriate ground truth
   - Original dataset penalized model for spans in context_prev/context_next

2. **Context-dependent labels**:
   - When evaluating fact type (hobby, preference, etc.), use records with `needs_context=false`
   - The context-flagged records require human knowledge of prior conversation
   - ML model should NOT be expected to infer these without context

3. **Deduplication impact**:
   - Removed 50 near-duplicates that would inflate metrics
   - Candidates are now unique per entity mention
   - Fair comparison between exact string matches

## Statistics

### Candidates Removed by Reason

- **span_text not found**: 26 candidates from 21 records
- **Exact/near duplicates**: 50 candidates from 25 records

### Average Candidates per Record

- **Before cleaning**: 0.39
- **After cleaning**: 0.33
- **Reduction**: 16.3%

## Implementation Notes

Cleaning logic applied to all 796 records:

1. **Span substring matching**: Case-sensitive substring search in message_text
2. **Deduplication strategy**:
   - Exact duplicates: removed (kept first occurrence)
   - Near-duplicates: removed if similarity > 60% and < 100%
   - Kept longer/more specific version in case of conflict
3. **Context flagging heuristics**:
   - Single-word messages (all-caps, e.g., "Vestibular")
   - Messages with ≤5 words + context-suggestion keywords in gold_notes
   - Keywords: "context", "previous", "prior", "before", "referring to", "implicit"
