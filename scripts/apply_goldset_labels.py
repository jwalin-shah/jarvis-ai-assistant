"""Apply gold labels to fact_goldset_200.csv"""

import argparse
import csv
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from tqdm import tqdm

INPUT = "training_data/fact_goldset/fact_goldset_200.csv"
OUTPUT = "training_data/fact_goldset/fact_goldset_200.csv"

# Labels: (gold_keep, gold_fact_type, gold_subject, gold_subject_resolution, gold_anchor_message_id, gold_notes)
LABELS = {
    "fact_gs_0001": (0, "", "", "", "", "logistics; relaying message"),
    "fact_gs_0002": (0, "", "", "", "", "vague opinion"),
    "fact_gs_0003": (
        1,
        "work",
        "other_participant",
        "other_participant",
        "",
        "not currently working; follows coworkers on social media",
    ),
    "fact_gs_0004": (1, "health", "me", "speaker", "", "chronic pain; seeing neurologist"),
    "fact_gs_0005": (1, "preference", "me", "speaker", "", "likes hard massages"),
    "fact_gs_0006": (0, "", "", "", "", "logistics; offering ride"),
    "fact_gs_0007": (0, "", "", "", "", "transient; mom cooking today"),
    "fact_gs_0008": (
        1,
        "relationship",
        "other_participant",
        "other_participant",
        "",
        "has a brother",
    ),
    "fact_gs_0009": (0, "", "", "", "", "system message; reaction"),
    "fact_gs_0010": (0, "", "", "", "", "vague comparison"),
    "fact_gs_0011": (0, "", "", "", "", "sports/school opinion; not personal fact"),
    "fact_gs_0012": (0, "", "", "", "", "logistics"),
    "fact_gs_0013": (0, "", "", "", "", "logistics"),
    "fact_gs_0014": (0, "", "", "", "", "no fact"),
    "fact_gs_0015": (0, "", "", "", "", "figurative; fantasy basketball context"),
    "fact_gs_0016": (0, "", "", "", "", "logistics"),
    "fact_gs_0017": (0, "", "", "", "", "transient; essay opinion"),
    "fact_gs_0018": (
        1,
        "relationship",
        "other_participant",
        "other_participant",
        "",
        "has dad and brothers; dad plays more (games)",
    ),
    "fact_gs_0019": (0, "", "", "", "", "logistics; transient"),
    "fact_gs_0020": (
        1,
        "hobby",
        "other_participant",
        "other_participant",
        "",
        "can ski really well",
    ),
    "fact_gs_0021": (0, "", "", "", "", "no fact"),
    "fact_gs_0022": (0, "", "", "", "", "no fact; time reference"),
    "fact_gs_0023": (1, "relationship", "me", "speaker", "", "has feelings for someone"),
    "fact_gs_0024": (0, "", "", "", "", "transient; dad behind them"),
    "fact_gs_0025": (0, "", "", "", "", "no fact"),
    "fact_gs_0026": (0, "", "", "", "", "sports reaction"),
    "fact_gs_0027": (0, "", "", "", "", "time reference"),
    "fact_gs_0028": (0, "", "", "", "", "vague; describing Hawaii"),
    "fact_gs_0029": (0, "", "", "", "", "transient status"),
    "fact_gs_0030": (0, "", "", "", "", "acknowledgment"),
    "fact_gs_0031": (0, "", "", "", "", "no durable fact; about funny pics"),
    "fact_gs_0032": (0, "", "", "", "", "asking question; fact in context not message"),
    "fact_gs_0033": (0, "", "", "", "", "exclamation"),
    "fact_gs_0034": (
        1,
        "preference",
        "dad",
        "named_entity",
        "",
        "other participant's dad roots for Chiefs",
    ),
    "fact_gs_0035": (0, "", "", "", "", "empathy statement; no fact"),
    "fact_gs_0036": (0, "", "", "", "", "question fragment"),
    "fact_gs_0037": (0, "", "", "", "", "relaying mom's question"),
    "fact_gs_0038": (0, "", "", "", "", "vague; no fact"),
    "fact_gs_0039": (0, "", "", "", "", "transient location update"),
    "fact_gs_0040": (0, "", "", "", "", "playful; no fact"),
    "fact_gs_0041": (0, "", "", "", "", "reaction; no fact"),
    "fact_gs_0042": (0, "", "", "", "", "philosophical musing; not personal"),
    "fact_gs_0043": (0, "", "", "", "", "about job interview process; no personal fact"),
    "fact_gs_0044": (0, "", "", "", "", "transient travel; dad driving"),
    "fact_gs_0045": (0, "", "", "", "", "logistics"),
    "fact_gs_0046": (0, "", "", "", "", "logistics"),
    "fact_gs_0047": (0, "", "", "", "", "transient; coming home from class"),
    "fact_gs_0048": (0, "", "", "", "", "self-deprecation; not durable"),
    "fact_gs_0049": (0, "", "", "", "", "no fact"),
    "fact_gs_0050": (0, "", "", "", "", "reaction"),
    "fact_gs_0051": (0, "", "", "", "", "reaction"),
    "fact_gs_0052": (
        1,
        "preference",
        "me",
        "speaker",
        "",
        "likes chocolate and peanut butter; never had Reese's",
    ),
    "fact_gs_0053": (0, "", "", "", "", "no fact in main message; health facts in context only"),
    "fact_gs_0054": (
        1,
        "preference",
        "other_participant",
        "other_participant",
        "",
        "likes 100T esports team",
    ),
    "fact_gs_0055": (0, "", "", "", "", "apartment logistics"),
    "fact_gs_0056": (0, "", "", "", "", "fantasy football reaction"),
    "fact_gs_0057": (0, "", "", "", "", "greeting; no fact"),
    "fact_gs_0058": (1, "hobby", "other_participant", "other_participant", "", "loves reading"),
    "fact_gs_0059": (0, "", "", "", "", "goodnight"),
    "fact_gs_0060": (0, "", "", "", "", "asking about person; no fact in main message"),
    "fact_gs_0061": (0, "", "", "", "", "transient schedule; mom has work today"),
    "fact_gs_0062": (0, "", "", "", "", "scheduling; brother fact in context only"),
    "fact_gs_0063": (0, "", "", "", "", "asking about group membership"),
    "fact_gs_0064": (
        1,
        "relationship",
        "other_participant",
        "other_participant",
        "",
        "likes someone",
    ),
    "fact_gs_0065": (0, "", "", "", "", "vague; dad-in-RV fact is in context not main msg"),
    "fact_gs_0066": (
        1,
        "location",
        "me",
        "speaker",
        "",
        "moving in; mom will ship remaining stuff",
    ),
    "fact_gs_0067": (0, "", "", "", "", "vague; feet got wet is transient"),
    "fact_gs_0068": (0, "", "", "", "", "literal filler text"),
    "fact_gs_0069": (0, "", "", "", "", "joking expression"),
    "fact_gs_0070": (0, "", "", "", "", "bot/automated delivery message"),
    "fact_gs_0071": (0, "", "", "", "", "scheduling around practice; no explicit durable fact"),
    "fact_gs_0072": (0, "", "", "", "", "sports commentary"),
    "fact_gs_0073": (0, "", "", "", "", "reaction"),
    "fact_gs_0074": (0, "", "", "", "", "transient logistics; dad can drop off"),
    "fact_gs_0075": (
        1,
        "preference",
        "me",
        "speaker",
        "",
        "likes the city a lot; prefers UTD as school",
    ),
    "fact_gs_0076": (0, "", "", "", "", "transient purchase decision"),
    "fact_gs_0077": (0, "", "", "", "", "asking about TV service"),
    "fact_gs_0078": (0, "", "", "", "", "logistics; showing something to Sid"),
    "fact_gs_0079": (0, "", "", "", "", "fantasy football frustration"),
    "fact_gs_0080": (
        1,
        "other",
        "me",
        "speaker",
        "",
        "celebrates Rakshabandhan; has street play practice",
    ),
    "fact_gs_0081": (0, "", "", "", "", "sports commentary"),
    "fact_gs_0082": (0, "", "", "", "", "joking"),
    "fact_gs_0083": (0, "", "", "", "", "greeting"),
    "fact_gs_0084": (0, "", "", "", "", "logistics"),
    "fact_gs_0085": (0, "", "", "", "", "phone deal logistics"),
    "fact_gs_0086": (0, "", "", "", "", "sports acknowledgment"),
    "fact_gs_0087": (
        1,
        "preference",
        "other_participant",
        "other_participant",
        "",
        "likes grammar; only knows one language",
    ),
    "fact_gs_0088": (0, "", "", "", "", "transient; school format speculation"),
    "fact_gs_0089": (0, "", "", "", "", "gaming stats"),
    "fact_gs_0090": (0, "", "", "", "", "no fact"),
    "fact_gs_0091": (
        0,
        "",
        "",
        "",
        "",
        "main msg is just intent; shrooms/edibles facts in context",
    ),
    "fact_gs_0092": (
        1,
        "other",
        "me",
        "speaker",
        "",
        "family uses Sprint; dad comparing phone plans",
    ),
    "fact_gs_0093": (0, "", "", "", "", "joke about address"),
    "fact_gs_0094": (0, "", "", "", "", "thanks"),
    "fact_gs_0095": (
        1,
        "relationship",
        "me",
        "speaker",
        "",
        "closest friend is the recipient; prefers talking to them over others",
    ),
    "fact_gs_0096": (0, "", "", "", "", "asking about railroad; subleasing in context"),
    "fact_gs_0097": (0, "", "", "", "", "greeting"),
    "fact_gs_0098": (0, "", "", "", "", "reaction to beer price"),
    "fact_gs_0099": (0, "", "", "", "", "transient; visiting old friends"),
    "fact_gs_0100": (1, "other", "me", "speaker", "", "celebrates Diwali"),
    "fact_gs_0101": (0, "", "", "", "", "question; no fact"),
    "fact_gs_0102": (0, "", "", "", "", "transient; studying rn"),
    "fact_gs_0103": (
        0,
        "",
        "",
        "",
        "",
        "reciprocating love; close relationship implied but no new fact",
    ),
    "fact_gs_0104": (0, "", "", "", "", "well-wishing; transient"),
    "fact_gs_0105": (0, "", "", "", "", "cooking instruction; gym/lab facts in context only"),
    "fact_gs_0106": (1, "education", "me", "speaker", "", "hates UTD (their school)"),
    "fact_gs_0107": (0, "", "", "", "", "transient phone logistics"),
    "fact_gs_0108": (0, "", "", "", "", "travel time estimation"),
    "fact_gs_0109": (0, "", "", "", "", "sports exclamation"),
    "fact_gs_0110": (0, "", "", "", "", "transient; stuck at home during COVID"),
    "fact_gs_0111": (
        1,
        "health",
        "me",
        "speaker",
        "",
        "mental health struggles; head pain recurring",
    ),
    "fact_gs_0112": (0, "", "", "", "", "referring to someone; no durable fact"),
    "fact_gs_0113": (0, "", "", "", "", "listing song names"),
    "fact_gs_0114": (0, "", "", "", "", "time; fantasy league drafting in context"),
    "fact_gs_0115": (0, "", "", "", "", "transient schedule; physics class at specific time"),
    "fact_gs_0116": (0, "", "", "", "", "sports commentary"),
    "fact_gs_0117": (0, "", "", "", "", "reaction"),
    "fact_gs_0118": (
        1,
        "other",
        "other_participant",
        "other_participant",
        "",
        "dad took the Civic to work; family owns Honda Civic",
    ),
    "fact_gs_0119": (0, "", "", "", "", "greeting"),
    "fact_gs_0120": (0, "", "", "", "", "compliment; no new fact"),
    "fact_gs_0121": (0, "", "", "", "", "transient; in Zoom"),
    "fact_gs_0122": (0, "", "", "", "", "transient; neighbor's ticket discount"),
    "fact_gs_0123": (0, "", "", "", "", "transient schedule; Sid gets off work at 7"),
    "fact_gs_0124": (0, "", "", "", "", "frisbee throw discussion"),
    "fact_gs_0125": (
        1,
        "health",
        "me",
        "speaker",
        "",
        "recovering from injury; exercise regimen 1min walk/1min run; dad involved in medical decisions; going to SLO and SB",
    ),
    "fact_gs_0126": (0, "", "", "", "", "arrival announcement"),
    "fact_gs_0127": (0, "", "", "", "", "exclamation"),
    "fact_gs_0128": (0, "", "", "", "", "mom wants some items; transient"),
    "fact_gs_0129": (0, "", "", "", "", "emotional support context"),
    "fact_gs_0130": (0, "", "", "", "", "game name"),
    "fact_gs_0131": (0, "", "", "", "", "suggesting food meetup"),
    "fact_gs_0132": (0, "", "", "", "", "wanting to join; trip planning in context"),
    "fact_gs_0133": (0, "", "", "", "", "asking about reading/books"),
    "fact_gs_0134": (0, "", "", "", "", "something not working"),
    "fact_gs_0135": (0, "", "", "", "", "encouragement; Adi cooks is in context"),
    "fact_gs_0136": (0, "", "", "", "", "phone acting up"),
    "fact_gs_0137": (0, "", "", "", "", "dad deciding; vague"),
    "fact_gs_0138": (0, "", "", "", "", "acknowledgment"),
    "fact_gs_0139": (0, "", "", "", "", "momentary sports trade opinion"),
    "fact_gs_0140": (0, "", "", "", "", "observation about clothing"),
    "fact_gs_0141": (0, "", "", "", "", "basketball trade opinion"),
    "fact_gs_0142": (0, "", "", "", "", "sports prediction"),
    "fact_gs_0143": (0, "", "", "", "", "joke about dropping out"),
    "fact_gs_0144": (0, "", "", "", "", "edible/mom concern in context only"),
    "fact_gs_0145": (0, "", "", "", "", "event ticket logistics"),
    "fact_gs_0146": (0, "", "", "", "", "asking what NAPLEX is"),
    "fact_gs_0147": (0, "", "", "", "", "transient; likes own clothes"),
    "fact_gs_0148": (0, "", "", "", "", "contextual; mom protective"),
    "fact_gs_0149": (0, "", "", "", "", "logistics; check remaining"),
    "fact_gs_0150": (0, "", "", "", "", "reaction"),
    "fact_gs_0151": (0, "", "", "", "", "food at gathering; dad's sauce is transient"),
    "fact_gs_0152": (0, "", "", "", "", "reaction"),
    "fact_gs_0153": (0, "", "", "", "", "transient; mom knew groom's family at specific event"),
    "fact_gs_0154": (0, "", "", "", "", "asking about arrival"),
    "fact_gs_0155": (0, "", "", "", "", "birthday mention; transient"),
    "fact_gs_0156": (0, "", "", "", "", "car purchase discussion; not yet bought"),
    "fact_gs_0157": (0, "", "", "", "", "cancelled travel plan"),
    "fact_gs_0158": (0, "", "", "", "", "ticket budget; Mavs fan implicit from group"),
    "fact_gs_0159": (0, "", "", "", "", "transient schedule"),
    "fact_gs_0160": (0, "", "", "", "", "transient location"),
    "fact_gs_0161": (0, "", "", "", "", "miscommunication"),
    "fact_gs_0162": (0, "", "", "", "", "supportive statement; not factual"),
    "fact_gs_0163": (0, "", "", "", "", "reaction; withdrawal/ultimate facts in context only"),
    "fact_gs_0164": (
        1,
        "other",
        "me",
        "speaker",
        "",
        "personality: tends to repress emotions and feelings",
    ),
    "fact_gs_0165": (0, "", "", "", "", "transient; dad's reaction to game"),
    "fact_gs_0166": (0, "", "", "", "", "no fact"),
    "fact_gs_0167": (0, "", "", "", "", "apartment lease terms"),
    "fact_gs_0168": (0, "", "", "", "", "vague dislike"),
    "fact_gs_0169": (0, "", "", "", "", "reaction; sister fact in context"),
    "fact_gs_0170": (
        1,
        "health",
        "me",
        "speaker",
        "",
        "ongoing head pressure; plays ultimate and basketball",
    ),
    "fact_gs_0171": (1, "preference", "me", "speaker", "", "never had spicy palak paneer"),
    "fact_gs_0172": (0, "", "", "", "", "reaction to photo"),
    "fact_gs_0173": (0, "", "", "", "", "suggesting someone for tech help"),
    "fact_gs_0174": (0, "", "", "", "", "coworker bought a car; tangential"),
    "fact_gs_0175": (0, "", "", "", "", "acknowledgment"),
    "fact_gs_0176": (0, "", "", "", "", "encouragement"),
    "fact_gs_0177": (1, "other", "me", "speaker", "", "uses brother's bike"),
    "fact_gs_0178": (1, "preference", "me", "speaker", "", "likes most movies"),
    "fact_gs_0179": (0, "", "", "", "", "schedule confusion"),
    "fact_gs_0180": (0, "", "", "", "", "reaction"),
    "fact_gs_0181": (
        1,
        "hobby",
        "me",
        "speaker",
        "",
        "wants to learn guitar; enjoys strumming random chords",
    ),
    "fact_gs_0182": (
        1,
        "other",
        "other_participant",
        "other_participant",
        "",
        "getting iPhone X from parent (mama); dad flying it over",
    ),
    "fact_gs_0183": (0, "", "", "", "", "vague; about reading content"),
    "fact_gs_0184": (0, "", "", "", "", "typo correction"),
    "fact_gs_0185": (
        1,
        "work",
        "other_participant",
        "other_participant",
        "",
        "has a job; picking up friends from airport",
    ),
    "fact_gs_0186": (
        1,
        "preference",
        "other_participant",
        "other_participant",
        "",
        "likes Handsome Owl and James and Giant Peach sandwiches",
    ),
    "fact_gs_0187": (
        0,
        "",
        "",
        "",
        "",
        "focal message 'from the mainland' too weak; beach house fact from context only",
    ),
    "fact_gs_0188": (
        1,
        "hobby",
        "other_participant",
        "other_participant",
        "",
        "does yoga; physical issues needing stretching",
    ),
    "fact_gs_0189": (0, "", "", "", "", "asking about a gift"),
    "fact_gs_0190": (0, "", "", "", "", "gaming joke"),
    "fact_gs_0191": (0, "", "", "", "", "nevermind"),
    "fact_gs_0192": (0, "", "", "", "", "transient exam result"),
    "fact_gs_0193": (0, "", "", "", "", "basketball player comparison metaphor"),
    "fact_gs_0194": (0, "", "", "", "", "sports stat reference"),
    "fact_gs_0195": (0, "", "", "", "", "poll announcement"),
    "fact_gs_0196": (0, "", "", "", "", "sports observation"),
    "fact_gs_0197": (0, "", "", "", "", "one-time venue change; not regular attendance"),
    "fact_gs_0198": (0, "", "", "", "", "sports opinion about player"),
    "fact_gs_0199": (
        0,
        "",
        "",
        "",
        "",
        "reaction wrapper (Disliked); fact in quoted text not focal message",
    ),
    "fact_gs_0200": (0, "", "", "", "", "reaction to joke"),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=INPUT,
        help="Path to input goldset CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT,
        help="Path to output goldset CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print row progress every N rows when processing >10 rows (default: %(default)s).",
    )
    return parser.parse_args(argv)


def apply_labels(input_path: str, output_path: str, progress_every: int = 25) -> None:
    """Apply manual labels to the provided CSV file."""
    try:
        with open(input_path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)
    except OSError as exc:
        print(f"Error reading input CSV '{input_path}': {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc

    if not fieldnames:
        print(f"Error: input CSV '{input_path}' is missing headers.", file=sys.stderr, flush=True)
        raise SystemExit(1)

    total_rows = len(rows)
    if total_rows > 10:
        print(f"Applying labels to {total_rows} rows...", flush=True)

    for idx, row in enumerate(tqdm(rows, desc="Labeling", total=total_rows), 1):
        sid = row["sample_id"]
        if sid in LABELS:
            keep, ftype, subj, res, anchor, notes = LABELS[sid]
            row["gold_keep"] = str(keep)
            row["gold_fact_type"] = ftype
            row["gold_subject"] = subj
            row["gold_subject_resolution"] = res
            row["gold_anchor_message_id"] = anchor
            row["gold_notes"] = notes
        else:
            print(f"WARNING: No label for {sid}", file=sys.stderr, flush=True)

        if (
            total_rows > 10
            and progress_every > 0
            and (idx % progress_every == 0 or idx == total_rows)
        ):
            print(f"  processed {idx}/{total_rows} rows", flush=True)

    try:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        print(f"Error writing output CSV '{output_path}': {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc

    labeled = sum(1 for v in LABELS.values() if v[0] == 1)
    print(
        f"Done. {labeled}/{len(LABELS)} labeled as keep=1 ({labeled / len(LABELS) * 100:.0f}%)",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run script entrypoint."""
    # Setup logging with FileHandler + StreamHandler
    log_file = Path("apply_goldset_labels.log")
    file_handler = logging.FileHandler(log_file, mode="a")
    stream_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[file_handler, stream_handler],
    )
    logging.info("Starting apply_goldset_labels.py")

    args = parse_args(argv)
    apply_labels(args.input, args.output, args.progress_every)
    logging.info("Finished apply_goldset_labels.py")


if __name__ == "__main__":
    main()
