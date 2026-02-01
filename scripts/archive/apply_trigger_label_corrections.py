#!/usr/bin/env python3
"""Apply trigger label corrections from the audit to create corrected JSONL files.

Run with: uv run python -m scripts.apply_trigger_label_corrections
"""

import json
from pathlib import Path

# Corrections for data/trigger_training_full.jsonl
# Format: {line_number: {"old_label": X, "new_label": Y, "text_pattern": "start of text"}}
TRAINING_CORRECTIONS = {
    # Agent 1 corrections (lines 1-650) - key patterns
    3: {"old": "bad_news", "new": "statement", "text_start": "Bruh I'm trying to plan"},
    6: {"old": "bad_news", "new": "reaction", "text_start": "Fkin Rubio"},
    16: {"old": "good_news", "new": "statement", "text_start": "Yeah I'm doing pretty well"},
    22: {"old": "good_news", "new": "statement", "text_start": "See you just make me smile"},
    24: {"old": "greeting", "new": "statement", "text_start": "love u son"},
    25: {"old": "good_news", "new": "statement", "text_start": "I feel much better yay"},
    26: {"old": "request", "new": "statement", "text_start": "Before she gets a chance"},
    29: {"old": "invitation", "new": "statement", "text_start": "But I can def pick you up"},
    30: {"old": "invitation", "new": "yn_question", "text_start": "Thursday around 2 then"},
    41: {"old": "good_news", "new": "statement", "text_start": "mike thomas is back"},
    44: {"old": "bad_news", "new": "statement", "text_start": "They just extended shelter"},
    70: {"old": "greeting", "new": "statement", "text_start": "haven't heard from you"},
    71: {"old": "bad_news", "new": "statement", "text_start": "Not feeling that great"},
    101: {"old": "request", "new": "info_question", "text_start": "Like where the games running"},
    115: {"old": "bad_news", "new": "statement", "text_start": "Like breathing and coughing"},
    118: {"old": "bad_news", "new": "statement", "text_start": "I had to take my meds"},
    122: {"old": "good_news", "new": "statement", "text_start": "future doctor"},
    124: {"old": "good_news", "new": "statement", "text_start": "i just feel so much better"},
    128: {"old": "bad_news", "new": "statement", "text_start": "Also my im team got railed"},
    132: {"old": "good_news", "new": "statement", "text_start": "ok they will pay for parking"},
    135: {"old": "bad_news", "new": "statement", "text_start": "my entire teams dtd"},
    143: {"old": "bad_news", "new": "statement", "text_start": "Then at like 2:40"},
    144: {"old": "bad_news", "new": "statement", "text_start": "also our wifi tweakin"},
    152: {"old": "bad_news", "new": "reaction", "text_start": "I'm a fucking choker"},
    154: {"old": "bad_news", "new": "statement", "text_start": "Got worse for like 4 weeks"},
    156: {"old": "bad_news", "new": "statement", "text_start": "And was like 50 here"},
    157: {"old": "bad_news", "new": "statement", "text_start": "i'm so scared for my exam"},
    175: {"old": "request", "new": "statement", "text_start": "If it's raining"},
    399: {"old": "invitation", "new": "ack", "text_start": "i'm down for moves"},
    437: {"old": "good_news", "new": "yn_question", "text_start": "r u finally jelous tho"},
    462: {"old": "good_news", "new": "bad_news", "text_start": "we won't be able to have our giggle"},
    507: {"old": "good_news", "new": "request", "text_start": "Buddy can you grab Chipotle"},
    516: {"old": "good_news", "new": "ack", "text_start": "ok i won't"},
    528: {"old": "bad_news", "new": "statement", "text_start": "I was genuinely considering going to the ER"},
    535: {"old": "bad_news", "new": "ack", "text_start": "unfortunately"},
    537: {"old": "bad_news", "new": "good_news", "text_start": "Like I decided I wanna get faster"},
    543: {"old": "bad_news", "new": "ack", "text_start": "LMFAO unfortunately"},
    548: {"old": "bad_news", "new": "info_question", "text_start": "Why I lost"},
    581: {"old": "bad_news", "new": "ack", "text_start": "Lol unfortunately yes"},
    584: {"old": "bad_news", "new": "info_question", "text_start": "It seems that I lost this contact"},
    586: {"old": "bad_news", "new": "ack", "text_start": "No unfortunately"},
    591: {"old": "bad_news", "new": "yn_question", "text_start": "if i lost feelings one day"},
    599: {"old": "bad_news", "new": "ack", "text_start": "unfortunately"},
    614: {"old": "statement", "new": "bad_news", "text_start": "We lost to the fucking lakers"},
    615: {"old": "bad_news", "new": "reaction", "text_start": "that was so sad"},
    618: {"old": "bad_news", "new": "info_question", "text_start": "lmaoo wdym unfortunately"},
    627: {"old": "bad_news", "new": "reaction", "text_start": "yea so sad"},
    634: {"old": "bad_news", "new": "reaction", "text_start": "Yeah so sad"},
    636: {"old": "bad_news", "new": "ack", "text_start": "unfortunately"},
    642: {"old": "bad_news", "new": "info_question", "text_start": "dang does it say why we lost"},
    646: {"old": "bad_news", "new": "info_question", "text_start": "Remember that quiz I thought I failed"},

    # Agent 2 corrections (lines 651-1273) - key patterns
    # Many greeting corrections
    1159: {"old": "greeting", "new": "yn_question", "text_start": "Yo u awake"},
    1163: {"old": "greeting", "new": "info_question", "text_start": "Yo wyd"},
    1168: {"old": "greeting", "new": "ack", "text_start": "Hey bro mb"},
    1169: {"old": "greeting", "new": "ack", "text_start": "Yo sorry"},
    1172: {"old": "greeting", "new": "info_question", "text_start": "Yo who tf is"},
    1175: {"old": "greeting", "new": "info_question", "text_start": "Yo where tf are you"},
    1182: {"old": "greeting", "new": "ack", "text_start": "hey thats fair"},
    1183: {"old": "greeting", "new": "ack", "text_start": "Yo yes"},
    1185: {"old": "greeting", "new": "statement", "text_start": "yo just got home"},
    1188: {"old": "greeting", "new": "ack", "text_start": "Yo sorry"},
    1193: {"old": "greeting", "new": "ack", "text_start": "Hey mb just saw"},
    1194: {"old": "greeting", "new": "yn_question", "text_start": "Yo dude u awake"},
    1199: {"old": "yn_question", "new": "greeting", "text_start": "How have you been"},
    1200: {"old": "yn_question", "new": "greeting", "text_start": "are you okay"},
    1203: {"old": "yn_question", "new": "info_question", "text_start": "What're u doin today"},
    1205: {"old": "yn_question", "new": "info_question", "text_start": "How u know"},
    1206: {"old": "yn_question", "new": "info_question", "text_start": "what's thursday"},
    1208: {"old": "yn_question", "new": "info_question", "text_start": "what've u been up to"},
    1224: {"old": "yn_question", "new": "info_question", "text_start": "What're u planning to do"},
    1226: {"old": "yn_question", "new": "request", "text_start": "can u invite blake out"},
    1228: {"old": "yn_question", "new": "info_question", "text_start": "when do u wanna work on it"},
    1231: {"old": "yn_question", "new": "request", "text_start": "do physics lab"},
    1076: {"old": "info_question", "new": "greeting", "text_start": "What's up"},
    1265: {"old": "info_question", "new": "greeting", "text_start": "what u doin"},

    # Invitation corrections
    980: {"old": "invitation", "new": "ack", "text_start": "Liked"},
    981: {"old": "invitation", "new": "yn_question", "text_start": "You free?"},
    983: {"old": "invitation", "new": "ack", "text_start": "Laughed at"},
    985: {"old": "invitation", "new": "statement", "text_start": "And I'm also tryna figure out"},
    986: {"old": "invitation", "new": "statement", "text_start": "Oh yeah I do wanna watch"},
    988: {"old": "invitation", "new": "ack", "text_start": "Loved"},
    992: {"old": "invitation", "new": "statement", "text_start": "Im trying to do UT Austin's"},
    993: {"old": "invitation", "new": "info_question", "text_start": "What time do you wanna go"},
    996: {"old": "invitation", "new": "request", "text_start": "Yes can you leave around"},
    997: {"old": "invitation", "new": "statement", "text_start": "Yeah dude I wanna come back"},
    998: {"old": "invitation", "new": "statement", "text_start": "I'm down to come with you"},
    1000: {"old": "invitation", "new": "statement", "text_start": "But like I just really wanna"},
    1003: {"old": "invitation", "new": "yn_question", "text_start": "are u free"},
    1007: {"old": "invitation", "new": "yn_question", "text_start": "wanna bet"},
    1012: {"old": "invitation", "new": "statement", "text_start": "lets get klay 50 again"},
    1017: {"old": "invitation", "new": "statement", "text_start": "Bruh I low key wanna go to Seattle"},
    1023: {"old": "invitation", "new": "ack", "text_start": "Liked"},
    1025: {"old": "invitation", "new": "info_question", "text_start": "y u asking me out"},
    1026: {"old": "invitation", "new": "yn_question", "text_start": "Are you free to do it"},
    1027: {"old": "invitation", "new": "request", "text_start": "Just say hey sorry"},

    # Request corrections
    951: {"old": "request", "new": "statement", "text_start": "Lmk if there's anything else"},
    960: {"old": "request", "new": "yn_question", "text_start": "Will u really turn up"},
    963: {"old": "request", "new": "ack", "text_start": "Loved"},
    965: {"old": "request", "new": "yn_question", "text_start": "will u still be at tutoring"},
    967: {"old": "request", "new": "invitation", "text_start": "Can u come tmrw"},
    968: {"old": "request", "new": "statement", "text_start": "unless it makes more sense"},
    969: {"old": "request", "new": "info_question", "text_start": "y u send me this"},
    972: {"old": "request", "new": "invitation", "text_start": "would you wanna do that"},
    977: {"old": "request", "new": "yn_question", "text_start": "Will you be eating dinner"},
    950: {"old": "request", "new": "yn_question", "text_start": "Will u eat Rajma"},
    1034: {"old": "yn_question", "new": "request", "text_start": "Can you take me to PT"},
    1035: {"old": "yn_question", "new": "request", "text_start": "Can u get me tmrw"},

    # Good news corrections (Agent 2)
    1119: {"old": "good_news", "new": "ack", "text_start": "Loved"},
    1120: {"old": "good_news", "new": "statement", "text_start": "and we won't be friends"},
    1122: {"old": "good_news", "new": "ack", "text_start": "Liked"},
    1124: {"old": "good_news", "new": "reaction", "text_start": "So happy chiefs are losing"},
    1126: {"old": "good_news", "new": "ack", "text_start": "Loved"},
    1128: {"old": "good_news", "new": "ack", "text_start": "Loved"},
    1129: {"old": "good_news", "new": "yn_question", "text_start": "U feeling better now"},
    1149: {"old": "good_news", "new": "greeting", "text_start": "Happy birthday"},
    1151: {"old": "good_news", "new": "bad_news", "text_start": "I am actually not feeling better"},
    1152: {"old": "good_news", "new": "info_question", "text_start": "Interview or got the job"},
    1157: {"old": "good_news", "new": "yn_question", "text_start": "Feeling better?"},

    # Reaction corrections
    1079: {"old": "reaction", "new": "yn_question", "text_start": "Did u see ur dads"},
    1082: {"old": "reaction", "new": "ack", "text_start": "omg slayyyy"},
    1087: {"old": "reaction", "new": "ack", "text_start": "Loved"},
    1089: {"old": "reaction", "new": "invitation", "text_start": "lets go to omg tacos"},
    1091: {"old": "reaction", "new": "statement", "text_start": "Idk wtf is up"},
    1093: {"old": "reaction", "new": "ack", "text_start": "Liked"},
    1094: {"old": "reaction", "new": "ack", "text_start": "Loved"},
    1098: {"old": "reaction", "new": "request", "text_start": "Yo wtf can I get the rundown"},
    1099: {"old": "reaction", "new": "yn_question", "text_start": "Also wait did you see this"},
    1107: {"old": "reaction", "new": "ack", "text_start": "Disliked"},
    1111: {"old": "reaction", "new": "greeting", "text_start": "How are you omg"},
    1114: {"old": "reaction", "new": "ack", "text_start": "nah wtf"},
    1115: {"old": "reaction", "new": "yn_question", "text_start": "Did you see hawks came back"},
}


def apply_corrections_to_training_file():
    """Apply corrections to the training file."""
    input_path = Path("data/trigger_training_full.jsonl")
    output_path = Path("results/trigger_training_full_corrected.jsonl")

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return

    corrections_applied = 0
    lines_processed = 0

    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for i, line in enumerate(f_in, start=1):
            lines_processed += 1
            data = json.loads(line.strip())

            # Check if this line has a correction
            if i in TRAINING_CORRECTIONS:
                correction = TRAINING_CORRECTIONS[i]
                text = data.get("text", "")

                # Verify the text matches (safety check)
                if text.startswith(correction["text_start"][:20]) or correction["text_start"][:10].lower() in text.lower():
                    if data["label"] == correction["old"]:
                        data["label"] = correction["new"]
                        data["notes"] = f"Corrected from {correction['old']} (audit)"
                        corrections_applied += 1

            f_out.write(json.dumps(data) + "\n")

    print(f"Training file: {corrections_applied} corrections applied out of {len(TRAINING_CORRECTIONS)} planned")
    print(f"Output written to: {output_path}")
    return corrections_applied


def apply_pattern_based_corrections(input_path: Path, output_path: Path):
    """Apply pattern-based corrections that work across any file."""
    corrections_applied = 0
    lines_processed = 0

    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            lines_processed += 1
            data = json.loads(line.strip())

            # Get text field (different field names in different files)
            text = data.get("text") or data.get("trigger_text", "")
            label = data.get("label")
            original_label = label

            if not text or not label:
                f_out.write(line)
                continue

            text_lower = text.lower().strip()

            # Pattern 1: Tapbacks should always be ack
            if text_lower.startswith(("liked ", "loved ", "laughed at ", "disliked ", "emphasized ", "questioned ")):
                if label != "ack":
                    data["label"] = "ack"
                    data["notes"] = f"Auto-corrected from {original_label} (tapback pattern)"
                    corrections_applied += 1

            # Pattern 2: Single word "unfortunately" is ack
            elif text_lower in ["unfortunately", "unfortunately."]:
                if label == "bad_news":
                    data["label"] = "ack"
                    data["notes"] = f"Auto-corrected from {original_label} (single word ack)"
                    corrections_applied += 1

            # Pattern 3: "How are you" variants are greeting
            elif text_lower in ["how are you", "how are you?", "how r u", "how r u?", "how are u",
                               "hows it going", "how's it going", "how's it going?",
                               "how have you been", "how have you been?", "how u been",
                               "how are you doing", "how are you doing?"]:
                if label == "info_question":
                    data["label"] = "greeting"
                    data["notes"] = f"Auto-corrected from {original_label} (social check-in)"
                    corrections_applied += 1

            # Pattern 4: "What's up" variants are greeting
            elif text_lower in ["what's up", "whats up", "what's up?", "whats up?",
                               "what's up??", "what's up???", "wassup", "sup"]:
                if label == "info_question":
                    data["label"] = "greeting"
                    data["notes"] = f"Auto-corrected from {original_label} (greeting pattern)"
                    corrections_applied += 1

            # Pattern 5: Brief reactions
            elif text_lower in ["yea so sad", "so sad", "yeah so sad", "that was so sad",
                               "thats so sad", "that's so sad"]:
                if label == "bad_news":
                    data["label"] = "reaction"
                    data["notes"] = f"Auto-corrected from {original_label} (brief reaction)"
                    corrections_applied += 1

            # Pattern 6: URLs/links are statements
            elif text_lower.startswith(("http://", "https://", "www.")):
                if label in ["yn_question", "info_question", "request", "invitation"]:
                    data["label"] = "statement"
                    data["notes"] = f"Auto-corrected from {original_label} (URL is statement)"
                    corrections_applied += 1

            f_out.write(json.dumps(data) + "\n")

    print(f"{input_path.name}: {corrections_applied} pattern-based corrections applied")
    return corrections_applied


def main():
    """Run all corrections."""
    print("=" * 60)
    print("Trigger Label Correction Script")
    print("=" * 60)

    # 1. Apply line-specific corrections to training file
    print("\n1. Applying line-specific corrections to training file...")
    training_corrections = apply_corrections_to_training_file()

    # 2. Apply pattern-based corrections to both files
    print("\n2. Applying pattern-based corrections...")

    training_path = Path("results/trigger_training_full_corrected.jsonl")
    if training_path.exists():
        # Apply pattern corrections on top of line corrections
        temp_path = Path("results/trigger_training_full_corrected_temp.jsonl")
        pattern_corrections = apply_pattern_based_corrections(training_path, temp_path)
        temp_path.rename(training_path)

    candidates_path = Path("results/trigger_candidates_labeled.jsonl")
    if candidates_path.exists():
        output_path = Path("results/trigger_candidates_labeled_corrected.jsonl")
        apply_pattern_based_corrections(candidates_path, output_path)

    print("\n" + "=" * 60)
    print("Corrections complete!")
    print("Output files:")
    print("  - results/trigger_training_full_corrected.jsonl")
    print("  - results/trigger_candidates_labeled_corrected.jsonl")
    print("=" * 60)


if __name__ == "__main__":
    main()
