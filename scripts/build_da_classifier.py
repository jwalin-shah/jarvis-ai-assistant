#!/usr/bin/env python3
"""
Build dialogue act classifiers for trigger and response types.

Two classifiers:
1. TRIGGER classifier - What type of response does the incoming message need?
2. RESPONSE classifier - What type is this response?

Approach:
- Use SWDA dataset as labeled exemplars
- Embed all exemplars with bge-small (semantic similarity)
- For new text, find k-nearest exemplars and vote on label
- Validate with adversarial test pairs (same meaning, different words)

Usage:
    uv run python -m scripts.build_da_classifier --build      # Build both indices
    uv run python -m scripts.build_da_classifier --validate   # Run validation suite
    uv run python -m scripts.build_da_classifier --test       # Interactive testing
    uv run python -m scripts.build_da_classifier --label      # Label our data
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# =============================================================================
# TAXONOMY DEFINITIONS
# =============================================================================

# TRIGGER CATEGORIES - What type of response does this message need?
TRIGGER_CATEGORIES = {
    "INVITATION": "Asking someone to do something together",
    "YN_QUESTION": "Yes/no question (not an invitation)",
    "WH_QUESTION": "Question asking for specific information (what/where/when/who/how)",
    "INFO_STATEMENT": "Sharing factual information",
    "OPINION": "Sharing opinion or feeling",
    "REQUEST": "Asking someone to do something for you",
    "GOOD_NEWS": "Positive announcement",
    "BAD_NEWS": "Negative announcement",
    "GREETING": "Opening a conversation",
    "ACKNOWLEDGE": "Acknowledging something (often ends conversation)",
}

# RESPONSE CATEGORIES - What type of response is this?
RESPONSE_CATEGORIES = {
    "AGREE": "Positive acceptance, yes, affirmation",
    "DECLINE": "Rejection, no, can't do it",
    "DEFER": "Non-committal, need to check, maybe",
    "ACKNOWLEDGE": "Simple confirmation without commitment (ok, got it)",
    "ANSWER": "Provides specific requested information",
    "QUESTION": "Asks for more information or clarification",
    "REACT_POSITIVE": "Excited, happy, congratulatory reaction",
    "REACT_SYMPATHY": "Supportive, sorry, sympathetic reaction",
    "STATEMENT": "Shares information (not answering a question)",
    "GREETING": "Greeting response",
}

# MAPPING: Trigger type → Valid response types
TRIGGER_TO_VALID_RESPONSES = {
    "INVITATION": ["AGREE", "DECLINE", "DEFER", "QUESTION"],
    "YN_QUESTION": ["AGREE", "DECLINE", "DEFER", "ANSWER", "QUESTION"],
    "WH_QUESTION": ["ANSWER", "DEFER", "QUESTION"],
    "INFO_STATEMENT": ["ACKNOWLEDGE", "REACT_POSITIVE", "REACT_SYMPATHY", "QUESTION", "STATEMENT"],
    "OPINION": ["AGREE", "DECLINE", "ACKNOWLEDGE", "QUESTION"],
    "REQUEST": ["AGREE", "DECLINE", "DEFER", "QUESTION"],
    "GOOD_NEWS": ["REACT_POSITIVE", "QUESTION", "ACKNOWLEDGE"],
    "BAD_NEWS": ["REACT_SYMPATHY", "QUESTION", "ACKNOWLEDGE"],
    "GREETING": ["GREETING", "QUESTION", "ACKNOWLEDGE"],
    "ACKNOWLEDGE": ["ACKNOWLEDGE", "QUESTION", "STATEMENT"],
}

# =============================================================================
# SWDA TAG MAPPINGS
# =============================================================================

# Map SWDA tags to TRIGGER categories
# These are utterances that PROMPT a response
SWDA_TO_TRIGGER = {
    # Questions that are invitations/proposals
    "qy": "YN_QUESTION",      # Yes-No-Question (some are invitations, handle below)
    "qy^d": "YN_QUESTION",    # Declarative Yes-No-Question
    "qw": "WH_QUESTION",      # Wh-Question
    "qw^d": "WH_QUESTION",    # Declarative Wh-Question
    "qo": "WH_QUESTION",      # Open-ended question
    "qh": "WH_QUESTION",      # Rhetorical question
    "qrr": "YN_QUESTION",     # Or-clause after yes-no
    "qr": "YN_QUESTION",      # Or-question

    # Statements
    "sd": "INFO_STATEMENT",   # Statement-non-opinion
    "sv": "OPINION",          # Statement-opinion
    "h": "OPINION",           # Hedge

    # Directives/Requests
    "ad": "REQUEST",          # Action-directive
    "co": "REQUEST",          # Offer/commit (often triggers response)

    # Greetings
    "fp": "GREETING",         # Conventional-opening
    "fc": "ACKNOWLEDGE",      # Conventional-closing

    # Acknowledgments (when they ack something, might need response)
    "b": "ACKNOWLEDGE",       # Backchannel
    "bk": "ACKNOWLEDGE",      # Response acknowledgment
    "ba": "ACKNOWLEDGE",      # Appreciation
}

# Map SWDA tags to RESPONSE categories
# These are utterances that RESPOND to something
SWDA_TO_RESPONSE = {
    # Agreement/Acceptance
    "aa": "AGREE",            # Agree/Accept
    "aap_am": "DEFER",        # Accept-part / Maybe
    "ny": "AGREE",            # Yes answers
    "na": "AGREE",            # Affirmative non-yes

    # Disagreement/Rejection
    "nn": "DECLINE",          # No answers
    "ar": "DECLINE",          # Reject
    "ng": "DECLINE",          # Negative non-no
    "arp_nd": "DECLINE",      # Dispreferred answers

    # Acknowledgment
    "b": "ACKNOWLEDGE",       # Backchannel
    "bk": "ACKNOWLEDGE",      # Response acknowledgment
    "ba": "REACT_POSITIVE",   # Appreciation (often positive reaction)
    "bh": "ACKNOWLEDGE",      # Backchannel in question form
    "bf": "ACKNOWLEDGE",      # Summarize/reformulate

    # Statements as responses
    "sd": "STATEMENT",        # Statement-non-opinion (as response = ANSWER or STATEMENT)
    "sv": "STATEMENT",        # Statement-opinion
    "no": "ANSWER",           # Other answers
    "h": "DEFER",             # Hedge

    # Questions as responses (clarification)
    "qy": "QUESTION",         # Yes-No-Question (as response = clarification)
    "qw": "QUESTION",         # Wh-Question (as response = clarification)
    "br": "QUESTION",         # Signal-non-understanding

    # Closings
    "fc": "ACKNOWLEDGE",      # Conventional-closing
    "ft": "REACT_POSITIVE",   # Thanking
    "fw": "GREETING",         # Welcome
    "fp": "GREETING",         # Conventional-opening
}

# =============================================================================
# MANUAL EXEMPLARS (to supplement SWDA with iMessage-specific patterns)
# =============================================================================

# Additional trigger exemplars for iMessage patterns not well-represented in SWDA
# EXPANDED: More examples for better semantic coverage
MANUAL_TRIGGER_EXEMPLARS = {
    "INVITATION": [
        # Direct invitations
        "Want to grab lunch?",
        "Down to hang?",
        "Are you coming tonight?",
        "You free Saturday?",
        "Let's do something",
        "We should hang out",
        "Wanna come over?",
        "You down?",
        "Want to join?",
        "Coming to the party?",
        "Let's get dinner",
        "Movie tonight?",
        "Drinks later?",
        "Want to go for a walk?",
        "Beach day tomorrow?",
        "Game night at mine?",
        # More variations
        "You coming or what?",
        "Let's grab coffee",
        "Wanna hang tomorrow?",
        "Should we meet up?",
        "Can you make it?",
        "Are you in?",
        "Join us?",
        "Come through?",
        "Pull up?",
        "You trying to come?",
        "Want to do something this weekend?",
        "Free for dinner?",
        "Got time to hang?",
        "Let's link",
        "We doing this or nah?",
    ],
    "YN_QUESTION": [
        # Factual yes/no (NOT invitations - just asking facts)
        "Did you see the game?",
        "Is it raining?",
        "Have you eaten?",
        "Did you get my message?",
        "Are you busy?",
        "Is everything okay?",
        "Did it work?",
        "Have you heard from them?",
        "Are you sure?",
        "Did you finish?",
        # More variations
        "You good?",
        "Everything alright?",
        "Did you make it?",
        "Is that true?",
        "You serious?",
        "Did you know about this?",
        "Have you tried it?",
        "Is it done?",
        "Did they respond?",
        "You saw that right?",
        "Is it open?",
        "Did you hear?",
        "You awake?",
        "Still there?",
        "Did I wake you?",
        "You ready?",
        "Is this right?",
        "Did you watch it?",
        "You remember?",
        "Is it working?",
        # More "Is it..." factual questions
        "Is it cold out?",
        "Is it cold?",
        "Is it hot?",
        "Is it raining out?",
        "Is it snowing?",
        "Is it crowded?",
        "Is it busy?",
        "Is it far?",
        "Is it close?",
        "Is it expensive?",
        "Is it good?",
        "Is it any good?",
        "Is it worth it?",
        "Is it true?",
        "Is it safe?",
        "Is it late?",
        "Is it early?",
        "Is it over?",
        "Is it starting?",
        "Is it cancelled?",
        # "Are you..." factual questions
        "Are you home?",
        "Are you there?",
        "Are you up?",
        "Are you coming?",
        "Are you close?",
        "Are you almost here?",
        "Are you working?",
        "Are you sleeping?",
        # "Did you..." / "Have you..." factual questions
        "Did you leave?",
        "Did you arrive?",
        "Did you get home?",
        "Did you find it?",
        "Have you left?",
        "Have you started?",
        "Have you been there?",
    ],
    "WH_QUESTION": [
        # Time questions (asking for INFO, not inviting)
        "What time?",
        "When should I come?",
        "When are you free?",
        "What time works?",
        "When does it start?",
        "How long will it take?",
        "When are you leaving?",
        "What time you getting here?",
        "When should I show up?",
        "What time should I be there?",
        "When is it?",
        "When do we meet?",
        "When are we meeting?",
        "When should we leave?",
        "How long is the drive?",
        "When does it end?",
        "What time is dinner?",
        "When is the reservation?",
        # Location questions
        "Where are we meeting?",
        "Where is it?",
        "What's the address?",
        "Where are you?",
        "Where should I go?",
        "What's the spot?",
        "Where at?",
        "Which place?",
        "Where do I park?",
        "Where should I meet you?",
        "What's the location?",
        "Where is everyone?",
        "Where should we go?",
        # Person questions (asking for INFO about attendees)
        "Who's coming?",
        "Who else is there?",
        "Who said that?",
        "Who are you with?",
        "Who's going?",
        "Who all is coming?",
        "Who else is going?",
        "Who's gonna be there?",
        "Who did you invite?",
        "Who's in?",
        "Who's all there?",
        "Who showed up?",
        "Who made it?",
        # What/how questions
        "What happened?",
        "How was it?",
        "What do you think?",
        "How much?",
        "Which one?",
        "What's the plan?",
        "How did it go?",
        "What should I bring?",
        "What are you doing?",
        "How are you getting there?",
        "What did they say?",
        "Why though?",
        "How come?",
        "What's going on?",
        "What do you mean?",
        "How so?",
        "What do I need to bring?",
        "What's the dress code?",
        "What are we doing?",
        "What's happening?",
        "What time should we leave?",
    ],
    "INFO_STATEMENT": [
        # Status updates (NOT requests - just sharing info)
        "Meeting moved to 3",
        "I'm running late",
        "Traffic is bad",
        "I'm here",
        "Just got home",
        "On my way",
        "Be there in 10",
        "I'll be late",
        "The restaurant is closed",
        "It's raining",
        "Flight got delayed",
        "Just parked",
        "Almost there",
        "Leaving now",
        "Got held up",
        "Running behind",
        "Just woke up",
        "Battery dying",
        "Phone was dead",
        "Just saw this",
        "They changed it to tomorrow",
        "It's canceled",
        "They're closed",
        "Got stuck at work",
        "Train is delayed",
        "Still at the office",
        # MORE status updates (commonly confused with REQUEST)
        "Heading there now",
        "Gonna be late",
        "Running a bit behind",
        "I'll be there soon",
        "Just left",
        "Leaving in 5",
        "About to head out",
        "Walking over now",
        "In an Uber",
        "Taking the train",
        "Driving now",
        "5 minutes away",
        "10 minutes out",
        "ETA 15 min",
        "Should be there by 7",
        "Probably gonna be late",
        "Might be a few minutes late",
        "Just got on the freeway",
        "In traffic rn",
        "Stuck in traffic",
        "Hit some traffic",
        "Taking longer than expected",
        "Be there shortly",
        "Coming now",
        "Omw",
        "Otw",
        # Location/status sharing
        "I'm at the restaurant",
        "At the bar",
        "Just got to the venue",
        "Waiting outside",
        "I'm inside",
        "By the front door",
        "Near the entrance",
        "At the back",
        "Found parking",
        "Looking for parking",
        "Circling the block",
        # Informational updates
        "Game starts at 8",
        "Doors open at 7",
        "It starts in an hour",
        "They close at 10",
        "Last call is at 1",
        "Happy hour ends at 6",
        "Reservation is at 7:30",
        "Our table is ready",
        "They have a wait",
        "It's packed",
        "Not too crowded",
        "Pretty empty actually",
        "Weather looks good",
        "Supposed to rain later",
        "It's cold out",
        "It's hot af",
    ],
    "OPINION": [
        "I think we should wait",
        "That movie was great",
        "I'm not sure about this",
        "Seems like a good idea",
        "I don't think that's right",
        "This place is amazing",
        "I love this",
        "Not a fan",
        "I prefer the other one",
        # More variations
        "I don't know about that",
        "Sounds sketchy",
        "That's kinda weird",
        "I'm feeling it",
        "Not really my thing",
        "That's actually pretty good",
        "I'm on the fence",
        "Hard to say",
        "I have mixed feelings",
        "Seems off to me",
    ],
    "REQUEST": [
        "Can you pick up milk?",
        "Send me the file",
        "Can you call me?",
        "Let me know when you're free",
        "Text me when you get there",
        "Can you help me with something?",
        "Pick me up at 5?",
        "Remind me later",
        "Can you check on that?",
        # More variations
        "Send me the address",
        "Forward me that email",
        "Can you cover for me?",
        "Give me a call",
        "Shoot me a text",
        "Can you look into it?",
        "Let me know what they say",
        "Keep me posted",
        "Send pics",
        "Share your location",
        "Can you grab that for me?",
        "Do me a favor?",
        "Can you handle this?",
        "Pass that along",
        "Fill me in later",
    ],
    "GOOD_NEWS": [
        "I got the job!",
        "We're engaged!",
        "I passed the exam!",
        "Guess what happened",
        "Great news!",
        "I did it!",
        "We won!",
        "It worked!",
        "I'm pregnant!",
        # More variations
        "You won't believe this",
        "Best day ever",
        "Finally!",
        "It actually happened",
        "I can't believe it",
        "They said yes!",
        "I made it",
        "We got approved",
        "They accepted me",
        "I'm officially done",
        "Closed the deal",
        "Got promoted!",
        "Passed my interview",
        "They loved it",
        "Everything worked out",
        "Such good news",
        "I'm so happy right now",
    ],
    "BAD_NEWS": [
        "We broke up",
        "I failed the test",
        "I got fired",
        "Bad news",
        "It didn't work out",
        "I'm not feeling well",
        "Something came up",
        "I have to cancel",
        "The deal fell through",
        # More variations
        "They rejected me",
        "I didn't get it",
        "Things went wrong",
        "It's over",
        "We lost",
        "They said no",
        "I messed up",
        "It fell apart",
        "Everything went sideways",
        "I'm struggling",
        "Having a rough day",
        "This sucks",
        "Not good",
        "I'm stressed",
        "Things aren't great",
        "I'm overwhelmed",
        "Got some bad news",
        "My car broke down",
        "I lost my wallet",
    ],
    "GREETING": [
        "Hey",
        "What's up",
        "Hi!",
        "Yo",
        "Hey there",
        "Hello",
        "Howdy",
        "Sup",
        "Hey hey",
        # More variations
        "Morning",
        "Good morning",
        "Evening",
        "Hiya",
        "What's good",
        "How's it going",
        "Hey you",
        "Heyyy",
        "Hellooo",
        "Ayy",
        # Informal greetings (commonly confused with acknowledge)
        "Supp",
        "Suppp",
        "Yooo",
        "Yoooo",
        "Ayyy",
        "Ayyyy",
        "Wassup",
        "Whaddup",
        "What up",
        "Whats good",
        "Heyy",
        "Hiii",
        "Hiiii",
        "G'morning",
        "Gm",
        "Good afternoon",
        "Good evening",
        "Evenin",
        "Night",
        "Good night",
        "Greetings",
        "Ello",
        "Oi",
        "Ahoy",
    ],
    "ACKNOWLEDGE": [
        "Ok",
        "Got it",
        "Sounds good",
        "Cool",
        "Alright",
        "K",
        "Perfect",
        "Works for me",
        "Noted",
        "Will do",
        # More variations
        "Copy",
        "Roger",
        "Understood",
        "Sure thing",
        "No problem",
        "All good",
        "Makes sense",
        "Fair enough",
        "I see",
        "Gotcha",
    ],
}

# Additional response exemplars for iMessage patterns
# EXPANDED: More examples for better semantic coverage
MANUAL_RESPONSE_EXEMPLARS = {
    "AGREE": [
        # Basic agreement
        "Yeah I'm down",
        "Sure",
        "Definitely",
        "For sure",
        "Sounds good",
        "I'm in",
        "Let's do it",
        "Yes!",
        "Absolutely",
        "Count me in",
        "Down",
        "Yep",
        "Yeah",
        "Sure thing",
        "I'd love to",
        # More variations
        "Yes please",
        "Of course",
        "I'm game",
        "Works for me",
        "Let's go",
        "I'm there",
        "Perfect",
        "That works",
        "Sounds great",
        "Deal",
        "Bet",
        "Say less",
        "You know it",
        "100%",
        "I'm so down",
        "Hell yes",
        "Yessss",
        "Totally",
        "Exactly",
        "That's right",
        "True",
        "Facts",
        "Agreed",
    ],
    "DECLINE": [
        # Basic decline
        "Can't make it",
        "Sorry I'm busy",
        "Not today",
        "I can't",
        "No thanks",
        "Nah",
        "I'll pass",
        "Not gonna work",
        "Rain check?",
        "Maybe next time",
        "Unfortunately no",
        "Won't be able to",
        # More variations
        "I wish I could",
        "Not this time",
        "I have plans",
        "Already committed",
        "Not feeling it",
        "I'm out",
        "Hard pass",
        "Gonna have to pass",
        "Not for me",
        "I'm good",
        "Nope",
        "That's a no from me",
        "I don't think so",
        "Probably not",
        "I really can't",
        "Sorry can't",
        "No can do",
        "Not happening",
        "I'll sit this one out",
    ],
    "DEFER": [
        # Basic defer
        "Maybe",
        "Let me check",
        "I'll see",
        "Not sure yet",
        "Possibly",
        "Let me get back to you",
        "I'll let you know",
        "Depends",
        "Might be able to",
        "Need to check my calendar",
        "I'll think about it",
        "We'll see",
        # More variations
        "Give me a sec",
        "Let me see",
        "I'll check and get back",
        "Gotta see if I'm free",
        "Hold on let me check",
        "Not sure if I can",
        "TBD",
        "I'll confirm later",
        "Possibly yeah",
        "Maybe I can swing it",
        "Let me figure it out",
        "I need to check first",
        "Might work",
        "Could be",
        "I'll try",
        "Hopefully",
    ],
    "ACKNOWLEDGE": [
        # Basic acknowledge
        "Ok",
        "Got it",
        "Cool",
        "Sounds good",
        "Alright",
        "K",
        "Perfect",
        "Makes sense",
        "Noted",
        "Word",
        "Bet",
        "Copy that",
        # More variations
        "Understood",
        "Roger",
        "Copy",
        "All good",
        "No worries",
        "Fair enough",
        "I see",
        "Gotcha",
        "Okay cool",
        "Ah okay",
        "Oh I see",
        "Right",
        "Ohh",
        "Ah gotcha",
        "Ohhh okay",
        "That makes sense",
        "Aight",
        "Kk",
        "Okok",
    ],
    "ANSWER": [
        # Time answers
        "2pm works",
        "Around 5",
        "Tomorrow",
        "In an hour",
        "Later tonight",
        "Next week",
        "After work",
        "In like 20 minutes",
        "Noon",
        "7ish",
        "Saturday afternoon",
        "Whenever you're free",
        "This evening",
        "In the morning",
        "Right now",
        "Give me 30 min",
        "Tonight",
        "Like 6pm",
        # Location answers
        "At the coffee shop",
        "The one on Main St",
        "Downtown",
        "My place",
        "At the park",
        "The usual spot",
        "Near the train station",
        "By the mall",
        "At home",
        "The address is...",
        "I'll send you the location",
        # People answers
        "Just me",
        "John and Sarah",
        "Everyone",
        "Just us two",
        "The whole crew",
        "A few people",
        "Same group as last time",
        # Quantity/price answers
        "About an hour",
        "$20",
        "The blue one",
        "Like 10 bucks",
        "Not much",
        "A couple hours",
        "The first one",
        "Either works",
        # General answers
        "Nothing much",
        "The usual",
        "Same as before",
        "Whatever works",
        "I don't mind",
        "Your call",
        "Up to you",
    ],
    "QUESTION": [
        # Clarification questions
        "What time?",
        "Where?",
        "Who else is coming?",
        "Which one?",
        "What do you mean?",
        "Like what?",
        "How so?",
        "What happened?",
        "Are you sure?",
        "Really?",
        # More variations
        "Wdym?",
        "Huh?",
        "Wait what?",
        "What?",
        "How come?",
        "Why?",
        "Since when?",
        "For real?",
        "You serious?",
        "What's going on?",
        "Which place?",
        "What should I bring?",
        "What's the plan?",
        "Who's going?",
        "How much?",
        "When though?",
        "Where at?",
        "How long?",
    ],
    "REACT_POSITIVE": [
        # Excitement
        "That's awesome!",
        "Congrats!",
        "No way!",
        "Amazing!",
        "So happy for you!",
        "Let's go!",
        "Hell yeah!",
        "That's great!",
        "Love that",
        "Yay!",
        "Finally!",
        "OMG",
        # More variations
        "Ayyy!",
        "Woohoo!",
        "That's incredible!",
        "I'm so proud of you",
        "You deserve it",
        "Well done!",
        "That's huge!",
        "Wow!",
        "Sick!",
        "Dope!",
        "Nicee",
        "Fire!",
        "W",
        "Big W",
        "LFG!",
        "Yes!!",
        "That's what I'm talking about!",
        "Killed it!",
        "So exciting!",
        "I knew you could do it",
        "You did it!",
        "That's amazing news",
    ],
    "REACT_SYMPATHY": [
        # Sympathy
        "I'm sorry",
        "That sucks",
        "Here for you",
        "That's rough",
        "I'm so sorry to hear that",
        "Sending love",
        "Let me know if you need anything",
        "Damn",
        "That's terrible",
        "I'm here if you need to talk",
        # More variations
        "I'm sorry to hear that",
        "That's awful",
        "I feel you",
        "Ugh that's the worst",
        "I'm here for you",
        "Thinking of you",
        "That really sucks",
        "I hate that for you",
        "Sending hugs",
        "Take care of yourself",
        "You okay?",
        "Wanna talk about it?",
        "I'm so sorry",
        "That's not fair",
        "Hang in there",
        "It'll get better",
        "I've been there",
        "That's hard",
        "You got this though",
        "Let me know how I can help",
    ],
    "STATEMENT": [
        # Status updates
        "I'm running late",
        "On my way",
        "Just got here",
        "Traffic was bad",
        "I'll be there soon",
        "Just saw this",
        "I was thinking the same thing",
        "Same here",
        "I agree",
        # More variations
        "I'm leaving now",
        "Almost there",
        "Just parked",
        "Running behind",
        "Got stuck in traffic",
        "Be there in 5",
        "Just woke up",
        "I'm up",
        "I'm home",
        "Left already",
        "I know right",
        "Exactly what I was thinking",
        "That's what I thought",
        "I feel the same way",
        "Me too",
        "Same tbh",
        "I was just about to say that",
        "That's fair",
        "Good point",
    ],
    "GREETING": [
        "Hey!",
        "What's up",
        "Not much",
        "Hi!",
        "Hey there",
        "Yo",
        "Hello!",
        "Sup",
        # More variations
        "Heyyy",
        "Heyy",
        "Morning!",
        "Good morning",
        "Evening!",
        "What's good",
        "How's it going",
        "Not much, you?",
        "Nothing much wbu",
        "Just chilling",
        "Same old",
        "Living the dream",
    ],
}

# =============================================================================
# VALIDATION TEST SUITE
# =============================================================================

# Pairs that should get the SAME label (semantically similar, different words)
TRIGGER_SHOULD_MATCH = [
    # INVITATION pairs
    (("Want to grab lunch?", "Down to eat?"), "INVITATION"),
    (("Are you coming tonight?", "You gonna be there later?"), "INVITATION"),
    (("Let's hang out", "We should chill"), "INVITATION"),
    (("Wanna join?", "You down to come?"), "INVITATION"),
    (("Movie tonight?", "Want to watch something later?"), "INVITATION"),

    # WH_QUESTION pairs
    (("What time?", "When should I come?"), "WH_QUESTION"),
    (("Where are we meeting?", "What's the spot?"), "WH_QUESTION"),
    (("Who's coming?", "Who else will be there?"), "WH_QUESTION"),
    (("How much?", "What's the price?"), "WH_QUESTION"),

    # YN_QUESTION pairs
    (("Did you see the game?", "You catch the match?"), "YN_QUESTION"),
    (("Is it raining?", "It raining out?"), "YN_QUESTION"),
    (("Have you eaten?", "Did you eat yet?"), "YN_QUESTION"),

    # INFO_STATEMENT pairs
    (("I'm running late", "Gonna be late"), "INFO_STATEMENT"),
    (("Meeting moved to 3", "They changed the meeting to 3"), "INFO_STATEMENT"),
    (("On my way", "Heading there now"), "INFO_STATEMENT"),

    # REQUEST pairs
    (("Can you pick up milk?", "Grab some milk?"), "REQUEST"),
    (("Send me the file", "Can you share that doc?"), "REQUEST"),
    (("Let me know when you're free", "Tell me your availability"), "REQUEST"),

    # GREETING pairs
    (("Hey", "Hi"), "GREETING"),
    (("What's up", "Sup"), "GREETING"),
    (("Hello", "Hey there"), "GREETING"),
]

RESPONSE_SHOULD_MATCH = [
    # AGREE pairs
    (("Yeah I'm down", "Sure let's do it"), "AGREE"),
    (("Definitely", "For sure"), "AGREE"),
    (("I'm in", "Count me in"), "AGREE"),
    (("Sounds good", "Works for me"), "AGREE"),  # Could also be ACKNOWLEDGE

    # DECLINE pairs
    (("Can't make it", "Won't be able to"), "DECLINE"),
    (("Sorry I'm busy", "Not gonna work for me"), "DECLINE"),
    (("I'll pass", "Not today"), "DECLINE"),

    # DEFER pairs
    (("Maybe", "Possibly"), "DEFER"),
    (("Let me check", "Need to see"), "DEFER"),
    (("I'll let you know", "Let me get back to you"), "DEFER"),

    # REACT_POSITIVE pairs
    (("That's awesome!", "Amazing!"), "REACT_POSITIVE"),
    (("Congrats!", "So happy for you!"), "REACT_POSITIVE"),
    (("No way!", "OMG!"), "REACT_POSITIVE"),

    # REACT_SYMPATHY pairs
    (("I'm sorry", "That sucks"), "REACT_SYMPATHY"),
    (("That's rough", "That's terrible"), "REACT_SYMPATHY"),
    (("Here for you", "Let me know if you need anything"), "REACT_SYMPATHY"),
]

# Pairs that should NOT get the same label (share words but different meaning)
TRIGGER_SHOULD_NOT_MATCH = [
    (("What's up?", "What's up with the project?"), ("GREETING", "WH_QUESTION")),
    (("Can you come?", "Can you send me that?"), ("INVITATION", "REQUEST")),
    (("Are you free?", "Is that free?"), ("INVITATION", "YN_QUESTION")),
    (("How are you?", "How do we fix this?"), ("GREETING", "WH_QUESTION")),
]

RESPONSE_SHOULD_NOT_MATCH = [
    (("Sure", "Are you sure?"), ("AGREE", "QUESTION")),
    (("No way!", "No way I can make it"), ("REACT_POSITIVE", "DECLINE")),
    (("That's great", "That's what I was thinking"), ("REACT_POSITIVE", "STATEMENT")),
]

# =============================================================================
# CLASS BALANCING CONFIGURATION
# =============================================================================

# Problem: STATEMENT dominates at ~87% of SWDA exemplars, causing most responses
# to be classified as STATEMENT even when they're clearly something else.
#
# Two solutions available:
# 1. Balanced sampling (preferred) - Downsample majority classes to match minority
# 2. Weighted voting (fallback) - Apply weights during k-NN voting
#
# Balanced sampling is cleaner:
# - Simpler (no weights to tune)
# - Faster (smaller index)
# - k-NN semantics remain intuitive

# Target exemplars per class for balanced sampling
# Set to None to use min class size, or specify a number
BALANCED_TARGET_PER_CLASS = 500

# PROPORTIONAL targets - based on actual message distribution analysis
# Equal sampling (500/class) over-corrected: DECLINE/DEFER/AGREE were over-predicted
# because their 500 exemplars competed equally with ANSWER's 500.
#
# In reality, most responses ARE answers/info. Proportional sampling reflects this:
# - ANSWER is the "default" for explanations, info sharing, etc.
# - Commitment classes (AGREE/DECLINE/DEFER) are rarer
# - Reactions and questions are moderate
PROPORTIONAL_TARGETS = {
    "ANSWER": 2000,        # Largest - explanations, info, "what happened", etc.
    "STATEMENT": 800,      # Opinions, status updates (distinct from ANSWER)
    "ACKNOWLEDGE": 600,    # "Ok", "got it", "cool" - common but specific
    "QUESTION": 600,       # Clarifications - "what time?", "where?"
    "REACT_POSITIVE": 500, # "Nice!", "Congrats!", "Awesome!"
    "AGREE": 400,          # "Yes", "Sure", "I'm down" - commitment
    "DECLINE": 400,        # "Can't", "No", "Sorry" - commitment
    "DEFER": 300,          # "Maybe", "Let me check" - less common
    "REACT_SYMPATHY": 200, # "Sorry to hear", "That sucks" - less common
    "GREETING": 200,       # "Hey", "Hi" - small category
}

# Response classifier weights (used in kNN voting when balanced=False)
# Only used as fallback when balanced sampling isn't applied
RESPONSE_CLASS_WEIGHTS = {
    "STATEMENT": 0.3,      # Heavily downweight (87% of exemplars)
    "AGREE": 2.0,          # Upweight (only 2.1% of exemplars)
    "DECLINE": 3.0,        # Heavily upweight (only 0.5%)
    "DEFER": 3.0,          # Heavily upweight (only 0.4%)
    "ANSWER": 3.0,         # Heavily upweight (only 0.2%)
    "ACKNOWLEDGE": 1.5,    # Slightly upweight
    "QUESTION": 1.5,       # Slightly upweight
    "REACT_POSITIVE": 2.0, # Upweight
    "REACT_SYMPATHY": 2.0, # Upweight
    "GREETING": 1.0,       # Keep as-is (already small)
}

# Trigger classifier weights (less imbalanced but still tune)
TRIGGER_CLASS_WEIGHTS = {
    "INFO_STATEMENT": 0.5,  # Downweight (tends to dominate)
    "OPINION": 0.7,         # Slightly downweight
    "INVITATION": 1.5,      # Upweight commitment triggers
    "REQUEST": 1.5,         # Upweight commitment triggers
    "YN_QUESTION": 1.2,     # Slightly upweight
    "WH_QUESTION": 1.2,     # Slightly upweight
    "GOOD_NEWS": 1.5,       # Upweight reaction triggers
    "BAD_NEWS": 1.5,        # Upweight reaction triggers
    "GREETING": 1.0,
    "ACKNOWLEDGE": 1.0,
}


# =============================================================================
# CORE IMPLEMENTATION
# =============================================================================

CLASSIFIER_DIR = Path.home() / ".jarvis" / "da_classifiers"


def load_swda():
    """Load SWDA dataset from local CSV files (downloaded from GitHub)."""
    import csv
    import re

    swda_dir = Path.home() / ".jarvis" / "swda_raw" / "swda"

    if not swda_dir.exists():
        print("  SWDA not found locally. Downloading from GitHub...")
        try:
            import subprocess
            cache_dir = Path.home() / ".jarvis" / "swda_raw"
            cache_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                "curl", "-sL",
                "https://raw.githubusercontent.com/cgpotts/swda/master/swda.zip",
                "-o", str(cache_dir / "swda.zip")
            ], check=True)
            subprocess.run([
                "unzip", "-q", "-o",
                str(cache_dir / "swda.zip"),
                "-d", str(cache_dir)
            ], check=True)
        except Exception as e:
            print(f"  Could not download SWDA: {e}")
            return {}

    print("Loading SWDA dataset from local CSV files...")

    # Find all CSV files
    csv_files = list(swda_dir.glob("sw*utt/*.csv"))
    print(f"  Found {len(csv_files)} conversation files")

    all_utterances = []
    for csv_file in csv_files:
        try:
            with open(csv_file, encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get("text", "")
                    act_tag = row.get("act_tag", "")

                    # Clean the text - remove disfluency markers and noise
                    text = re.sub(r"\{[A-Z] [^}]+\}", "", text)  # Remove {D ...}, {F ...}
                    text = re.sub(r"<[^>]+>", "", text)  # Remove <Noise>, etc.
                    text = re.sub(r"\[ [^\]]+ \+ \]", "", text)  # Remove [ ... + ]
                    text = re.sub(r"\[ ", "", text)
                    text = re.sub(r" \]", "", text)
                    text = re.sub(r"\s*/\s*$", "", text)  # Remove trailing /
                    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace

                    if text and len(text) >= 2 and act_tag:
                        all_utterances.append({
                            "text": text,
                            "act_tag": act_tag,
                        })
        except Exception:
            continue  # Skip problematic files

    print(f"  Loaded {len(all_utterances)} utterances total")

    # Return as single "train" split
    return {"train": all_utterances}


def extract_swda_exemplars(ds, mapping: dict, category_type: str) -> dict[str, list[str]]:
    """Extract exemplars from SWDA for given mapping."""
    exemplars = {cat: [] for cat in set(mapping.values())}

    for split in ["train", "validation", "test"]:
        if split not in ds:
            continue
        for item in ds[split]:
            text = item.get("text", "")
            act_tag = item.get("act_tag")

            # Skip empty or very short
            if not text or len(text.strip()) < 2:
                continue

            # Clean up text
            text = text.rstrip(" /").strip()
            if not text or len(text) < 2:
                continue

            # Map to our category
            if act_tag in mapping:
                category = mapping[act_tag]
                exemplars[category].append(text)

    # Print stats
    print(f"\n{category_type} exemplars from SWDA:")
    for cat, texts in sorted(exemplars.items()):
        print(f"  {cat:20} {len(texts):6}")

    return exemplars


def merge_exemplars(swda_exemplars: dict, manual_exemplars: dict) -> dict[str, list[str]]:
    """Merge SWDA and manual exemplars, deduplicating."""
    merged = {}

    for cat in set(list(swda_exemplars.keys()) + list(manual_exemplars.keys())):
        texts = set()
        texts.update(swda_exemplars.get(cat, []))
        texts.update(manual_exemplars.get(cat, []))
        merged[cat] = list(texts)

    return merged


def balance_exemplars(
    exemplars: dict[str, list[str]],
    target_per_class: int | None = None,
    min_per_class: int = 20,
    seed: int = 42,
    proportional: bool = False,
) -> dict[str, list[str]]:
    """Balance exemplars by downsampling majority classes.

    This is cleaner than weighted voting:
    - Simpler (no weights to tune)
    - Faster (smaller index)
    - k-NN semantics remain intuitive

    Args:
        exemplars: Dict mapping category -> list of texts.
        target_per_class: Target exemplars per class. If None, uses min class size.
            Ignored if proportional=True.
        min_per_class: Minimum class size to include (classes smaller are kept as-is).
        seed: Random seed for reproducibility.
        proportional: If True, use PROPORTIONAL_TARGETS instead of equal sampling.
            This gives larger classes more exemplars (e.g., ANSWER gets 2000,
            AGREE gets 400) to match realistic message distributions.

    Returns:
        Balanced exemplars dict.
    """
    import random

    random.seed(seed)

    class_sizes = {cat: len(texts) for cat, texts in exemplars.items()}

    if proportional:
        print("\nProportional sampling (using PROPORTIONAL_TARGETS)...")
        print(f"  Class sizes before: {class_sizes}")

        balanced = {}
        for cat, texts in exemplars.items():
            target = PROPORTIONAL_TARGETS.get(cat, 300)  # Default 300 for unknown
            if len(texts) <= target:
                balanced[cat] = texts.copy()
            else:
                balanced[cat] = random.sample(texts, target)

        new_sizes = {cat: len(texts) for cat, texts in balanced.items()}
        print(f"  Targets: {PROPORTIONAL_TARGETS}")
        print(f"  Class sizes after:  {new_sizes}")
        print(f"  Total: {sum(class_sizes.values())} -> {sum(new_sizes.values())}")

        return balanced

    # Original equal-sampling logic
    min_size = min(class_sizes.values())

    if target_per_class is None:
        target_per_class = min_size
    else:
        # Don't go below the smallest class
        target_per_class = min(target_per_class, max(class_sizes.values()))

    print(f"\nBalancing exemplars (target: {target_per_class} per class)...")
    print(f"  Class sizes before: {class_sizes}")

    balanced = {}
    for cat, texts in exemplars.items():
        if len(texts) <= target_per_class:
            # Keep all if class is small
            balanced[cat] = texts.copy()
        else:
            # Downsample
            balanced[cat] = random.sample(texts, target_per_class)

    new_sizes = {cat: len(texts) for cat, texts in balanced.items()}
    print(f"  Class sizes after:  {new_sizes}")
    print(f"  Total: {sum(class_sizes.values())} -> {sum(new_sizes.values())}")

    return balanced


def load_mined_exemplars(exemplars_dir: Path | None = None) -> dict[str, list[str]]:
    """Load additional exemplars mined from clusters and structural patterns.

    Args:
        exemplars_dir: Directory containing mined exemplars. Defaults to ~/.jarvis/da_exemplars/

    Returns:
        Dict mapping category -> list of texts.
    """
    if exemplars_dir is None:
        exemplars_dir = Path.home() / ".jarvis" / "da_exemplars"

    combined = {}

    # Load cluster-mined exemplars
    combined_file = exemplars_dir / "all_mined_exemplars.json"
    if combined_file.exists():
        with open(combined_file) as f:
            cluster_mined = json.load(f)
            for cat, texts in cluster_mined.items():
                combined[cat] = list(set(combined.get(cat, []) + texts))

    # Load structural-pattern-mined exemplars (Option C: better exemplars)
    structural_file = exemplars_dir / "structural_mined.json"
    if structural_file.exists():
        with open(structural_file) as f:
            structural_mined = json.load(f)
            for cat, texts in structural_mined.items():
                combined[cat] = list(set(combined.get(cat, []) + texts))
        print(f"  Loaded structural-mined exemplars from {structural_file}")

    return combined


def build_classifier_index(
    exemplars: dict[str, list[str]],
    name: str,
    batch_size: int = 500,
    balanced: bool = True,
    target_per_class: int | None = None,
    proportional: bool = False,
):
    """Build FAISS index for a classifier.

    Args:
        exemplars: Dict mapping category -> list of texts.
        name: Classifier name ('trigger' or 'response').
        batch_size: Batch size for embedding.
        balanced: If True, balance classes before building (recommended).
        target_per_class: Target exemplars per class for balancing.
        proportional: If True, use proportional sampling (PROPORTIONAL_TARGETS)
            instead of equal sampling. This is recommended for response classifier.
    """
    import faiss

    from jarvis.embedding_adapter import get_embedder

    # Apply balanced/proportional sampling if requested
    if balanced:
        if proportional:
            exemplars = balance_exemplars(exemplars, proportional=True)
        else:
            target = target_per_class or BALANCED_TARGET_PER_CLASS
            exemplars = balance_exemplars(exemplars, target_per_class=target)

    # Flatten exemplars with labels
    all_texts = []
    all_labels = []
    for category, texts in exemplars.items():
        for text in texts:
            all_texts.append(text)
            all_labels.append(category)

    print(f"\nBuilding {name} classifier with {len(all_texts)} exemplars...")

    # Embed using the unified embedder
    embedder = get_embedder()
    all_embeddings = []

    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i:i + batch_size]
        embeddings = embedder.encode(batch, normalize=True)
        all_embeddings.append(embeddings)
        print(f"  Embedded {min(i + batch_size, len(all_texts))}/{len(all_texts)}")

    embeddings_array = np.vstack(all_embeddings).astype(np.float32)

    # Build FAISS index (embeddings are already normalized)
    dim = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_array)

    # Save
    save_dir = CLASSIFIER_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(save_dir / "index.faiss"))
    np.save(save_dir / "labels.npy", np.array(all_labels))

    with open(save_dir / "texts.json", "w") as f:
        json.dump(all_texts, f)

    categories = list(exemplars.keys())
    metadata = {
        "count": len(all_texts),
        "dim": dim,
        "categories": categories,
        "category_counts": {cat: len(texts) for cat, texts in exemplars.items()},
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved to {save_dir}")

    return index, all_labels, all_texts


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    neighbors: list[dict] = None


class DialogueActClassifier:
    """Classify text using k-NN over exemplars with class weighting.

    The classifier uses class weights to counter class imbalance.
    STATEMENT dominates the training data (~87%), so it's downweighted
    while rarer but important classes (AGREE, DECLINE, DEFER) are upweighted.
    """

    def __init__(self, name: str, k: int = 10, use_class_weights: bool = True):
        import faiss

        self.name = name
        self.k = k
        self.use_class_weights = use_class_weights

        load_dir = CLASSIFIER_DIR / name
        self.index = faiss.read_index(str(load_dir / "index.faiss"))
        self.labels = np.load(load_dir / "labels.npy", allow_pickle=True)

        with open(load_dir / "texts.json") as f:
            self.texts = json.load(f)

        with open(load_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        # Select class weights based on classifier type
        if name == "response":
            self.class_weights = RESPONSE_CLASS_WEIGHTS
        elif name == "trigger":
            self.class_weights = TRIGGER_CLASS_WEIGHTS
        else:
            self.class_weights = {}

        from jarvis.embedding_adapter import get_embedder
        self.embedder = get_embedder()

    def _get_class_weight(self, label: str) -> float:
        """Get the weight for a class label."""
        if not self.use_class_weights:
            return 1.0
        return self.class_weights.get(label, 1.0)

    def classify(self, text: str, return_neighbors: bool = False) -> ClassificationResult:
        """Classify a single text with class-weighted voting."""

        # encode() returns normalized embeddings when normalize=True
        embedding = self.embedder.encode([text], normalize=True).astype(np.float32)

        scores, indices = self.index.search(embedding, self.k)

        neighbor_labels = [self.labels[i] for i in indices[0]]
        neighbor_texts = [self.texts[i] for i in indices[0]]
        neighbor_scores = scores[0].tolist()

        # Class-weighted voting: multiply similarity by class weight
        label_scores = Counter()
        for label, score in zip(neighbor_labels, neighbor_scores):
            weight = self._get_class_weight(label)
            label_scores[label] += score * weight

        best_label, best_score = label_scores.most_common(1)[0]

        # Confidence is based on proportion of weighted votes
        total_weighted_score = sum(label_scores.values())
        if total_weighted_score > 0:
            confidence = best_score / total_weighted_score
        else:
            confidence = sum(1 for lbl in neighbor_labels if lbl == best_label) / self.k

        neighbors = None
        if return_neighbors:
            neighbors = [
                {"text": t, "label": lbl, "score": s, "weight": self._get_class_weight(lbl)}
                for t, lbl, s in zip(neighbor_texts, neighbor_labels, neighbor_scores)
            ]

        return ClassificationResult(
            label=best_label,
            confidence=confidence,
            neighbors=neighbors,
        )

    def classify_batch(self, texts: list[str]) -> list[ClassificationResult]:
        """Classify multiple texts with class-weighted voting."""

        # encode() returns normalized embeddings when normalize=True
        embeddings = self.embedder.encode(texts, normalize=True).astype(np.float32)

        scores, indices = self.index.search(embeddings, self.k)

        results = []
        for i in range(len(texts)):
            neighbor_labels = [self.labels[idx] for idx in indices[i]]
            neighbor_scores = scores[i].tolist()

            # Class-weighted voting
            label_scores = Counter()
            for label, score in zip(neighbor_labels, neighbor_scores):
                weight = self._get_class_weight(label)
                label_scores[label] += score * weight

            best_label, best_score = label_scores.most_common(1)[0]

            # Confidence from weighted vote proportion
            total_weighted_score = sum(label_scores.values())
            if total_weighted_score > 0:
                confidence = best_score / total_weighted_score
            else:
                confidence = sum(1 for lbl in neighbor_labels if lbl == best_label) / self.k

            results.append(ClassificationResult(
                label=best_label,
                confidence=confidence,
            ))

        return results


# =============================================================================
# VALIDATION
# =============================================================================

def validate_classifier(
    classifier: DialogueActClassifier, should_match: list, should_not_match: list
):
    """Validate classifier with adversarial test pairs."""
    print(f"\n{'='*70}")
    print(f"VALIDATING: {classifier.name}")
    print(f"{'='*70}")

    # Test SHOULD_MATCH pairs
    print("\n--- Should Match (same meaning, different words) ---")
    match_correct = 0
    match_total = 0

    for (text1, text2), expected_label in should_match:
        result1 = classifier.classify(text1)
        result2 = classifier.classify(text2)

        # Check if both get the expected label
        both_correct = (result1.label == expected_label and result2.label == expected_label)
        same_label = (result1.label == result2.label)

        match_total += 1
        if both_correct:
            match_correct += 1
            status = "✓"
        elif same_label:
            status = "~"  # Same label but not expected
        else:
            status = "✗"

        print(f"  {status} Expected: {expected_label}")
        print(f"      '{text1}' → {result1.label} ({result1.confidence:.0%})")
        print(f"      '{text2}' → {result2.label} ({result2.confidence:.0%})")

    print(f"\n  Match accuracy: {match_correct}/{match_total} ({match_correct/match_total:.0%})")

    # Test SHOULD_NOT_MATCH pairs
    if should_not_match:
        print("\n--- Should NOT Match (share words, different meaning) ---")
        not_match_correct = 0
        not_match_total = 0

        for (text1, text2), (expected1, expected2) in should_not_match:
            result1 = classifier.classify(text1)
            result2 = classifier.classify(text2)

            # Check if they get different labels (and ideally the expected ones)
            different_labels = (result1.label != result2.label)
            correct_labels = (result1.label == expected1 and result2.label == expected2)

            not_match_total += 1
            if correct_labels:
                not_match_correct += 1
                status = "✓"
            elif different_labels:
                status = "~"  # Different but not expected
            else:
                status = "✗"  # Same label (bad)

            print(f"  {status} Expected: {expected1} vs {expected2}")
            print(f"      '{text1}' → {result1.label}")
            print(f"      '{text2}' → {result2.label}")

        print(f"\n  Different-label accuracy: {not_match_correct}/{not_match_total}")

    return match_correct / match_total if match_total > 0 else 0


def inspect_neighbors(classifier: DialogueActClassifier, text: str):
    """Show detailed neighbor information for a query."""
    result = classifier.classify(text, return_neighbors=True)

    print(f"\nQuery: '{text}'")
    print(f"Predicted: {result.label} (confidence: {result.confidence:.0%})")
    print(f"\nTop {len(result.neighbors)} neighbors:")

    for i, n in enumerate(result.neighbors):
        match = "✓" if n["label"] == result.label else " "
        print(f"  {i+1}. [{match}] '{n['text'][:50]}...' ({n['label']}, {n['score']:.3f})")


# =============================================================================
# MAIN COMMANDS
# =============================================================================

def cmd_build(balanced: bool = True, include_mined: bool = True, proportional: bool = False):
    """Build both classifier indices.

    Args:
        balanced: If True, balance classes by downsampling majority classes.
        include_mined: If True, include exemplars mined from clusters.
        proportional: If True, use proportional sampling instead of equal sampling.
            This gives ANSWER more exemplars than AGREE/DECLINE/DEFER, matching
            realistic message distributions.
    """
    ds = load_swda()

    # Load mined exemplars if available
    mined_exemplars = {}
    if include_mined:
        mined_exemplars = load_mined_exemplars()
        if mined_exemplars:
            print(f"\nLoaded mined exemplars: {list(mined_exemplars.keys())}")
        else:
            print("\nNo mined exemplars found (run mine_da_exemplars.py --extract first)")

    # Build TRIGGER classifier
    print("\n" + "="*70)
    print("BUILDING TRIGGER CLASSIFIER")
    print("="*70)

    swda_trigger = extract_swda_exemplars(ds, SWDA_TO_TRIGGER, "Trigger")
    trigger_exemplars = merge_exemplars(swda_trigger, MANUAL_TRIGGER_EXEMPLARS)

    print("\nAfter merging with manual exemplars:")
    for cat, texts in sorted(trigger_exemplars.items()):
        print(f"  {cat:20} {len(texts):6}")

    # Trigger classifier uses equal balancing (less imbalanced than response)
    build_classifier_index(trigger_exemplars, "trigger", balanced=balanced, proportional=False)

    # Build RESPONSE classifier
    print("\n" + "="*70)
    print("BUILDING RESPONSE CLASSIFIER")
    print("="*70)

    swda_response = extract_swda_exemplars(ds, SWDA_TO_RESPONSE, "Response")
    response_exemplars = merge_exemplars(swda_response, MANUAL_RESPONSE_EXEMPLARS)

    # Add mined exemplars for response types
    if mined_exemplars:
        print("\nAdding mined exemplars:")
        for da_type, texts in mined_exemplars.items():
            if da_type in response_exemplars:
                before = len(response_exemplars[da_type])
                response_exemplars[da_type] = list(
                    set(response_exemplars[da_type] + texts)
                )
                after = len(response_exemplars[da_type])
                print(f"  {da_type}: {before} -> {after} (+{after - before})")
            else:
                response_exemplars[da_type] = texts
                print(f"  {da_type}: 0 -> {len(texts)} (new)")

    print("\nAfter merging with manual + mined exemplars:")
    for cat, texts in sorted(response_exemplars.items()):
        print(f"  {cat:20} {len(texts):6}")

    # Response classifier uses proportional or equal balancing
    build_classifier_index(
        response_exemplars, "response", balanced=balanced, proportional=proportional
    )

    print("\n" + "="*70)
    print("BUILD COMPLETE")
    print("="*70)
    if balanced:
        if proportional:
            print("  Mode: PROPORTIONAL (classes sized according to PROPORTIONAL_TARGETS)")
        else:
            print("  Mode: BALANCED (classes downsampled to equal size)")
    else:
        print("  Mode: UNBALANCED (using class weights in voting)")
    print("="*70)


def cmd_validate():
    """Run validation suite on both classifiers."""
    trigger_clf = DialogueActClassifier("trigger")
    response_clf = DialogueActClassifier("response")

    trigger_acc = validate_classifier(
        trigger_clf, TRIGGER_SHOULD_MATCH, TRIGGER_SHOULD_NOT_MATCH
    )
    response_acc = validate_classifier(
        response_clf, RESPONSE_SHOULD_MATCH, RESPONSE_SHOULD_NOT_MATCH
    )

    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"  Trigger classifier: {trigger_acc:.0%} on semantic similarity pairs")
    print(f"  Response classifier: {response_acc:.0%} on semantic similarity pairs")


def cmd_test():
    """Interactive testing of classifiers."""
    trigger_clf = DialogueActClassifier("trigger")
    response_clf = DialogueActClassifier("response")

    test_triggers = [
        "Want to grab lunch?",
        "What time?",
        "Did you see the game?",
        "Meeting moved to 3pm",
        "I got the job!",
        "We broke up",
        "Hey what's up",
        "Can you pick up milk?",
        "I think we should wait",
    ]

    test_responses = [
        "Yeah I'm down",
        "Can't make it",
        "Maybe, let me check",
        "Ok got it",
        "2pm works",
        "What time?",
        "That's awesome!",
        "I'm sorry to hear that",
        "On my way",
    ]

    print("\n" + "="*70)
    print("TRIGGER CLASSIFICATION TEST")
    print("="*70)

    for text in test_triggers:
        inspect_neighbors(trigger_clf, text)

    print("\n" + "="*70)
    print("RESPONSE CLASSIFICATION TEST")
    print("="*70)

    for text in test_responses:
        inspect_neighbors(response_clf, text)

    # Show trigger → valid responses mapping
    print("\n" + "="*70)
    print("TRIGGER → VALID RESPONSE TYPES")
    print("="*70)

    for trigger_type, valid_responses in TRIGGER_TO_VALID_RESPONSES.items():
        print(f"  {trigger_type:20} → {', '.join(valid_responses)}")


def cmd_label(limit: int = 1000):
    """Label our extracted responses and triggers."""
    from jarvis.db import get_db

    trigger_clf = DialogueActClassifier("trigger")
    response_clf = DialogueActClassifier("response")

    db = get_db()
    pairs = db.get_training_pairs(min_quality=0.0)[:limit]

    # Pair is a dataclass, access attributes directly
    triggers = [p.trigger_text for p in pairs]
    responses = [p.response_text for p in pairs]

    print(f"\nLabeling {len(pairs)} pairs...")

    trigger_results = trigger_clf.classify_batch(triggers)
    response_results = response_clf.classify_batch(responses)

    # Analyze trigger distribution
    print("\n" + "="*50)
    print("TRIGGER TYPE DISTRIBUTION (our data)")
    print("="*50)

    trigger_counts = Counter(r.label for r in trigger_results)
    for label, count in trigger_counts.most_common():
        pct = count / len(trigger_results) * 100
        avg_conf = sum(r.confidence for r in trigger_results if r.label == label) / count
        print(f"  {label:20} {count:5} ({pct:5.1f}%)  avg_conf: {avg_conf:.2f}")

    # Analyze response distribution
    print("\n" + "="*50)
    print("RESPONSE TYPE DISTRIBUTION (our data)")
    print("="*50)

    response_counts = Counter(r.label for r in response_results)
    for label, count in response_counts.most_common():
        pct = count / len(response_results) * 100
        avg_conf = sum(r.confidence for r in response_results if r.label == label) / count
        print(f"  {label:20} {count:5} ({pct:5.1f}%)  avg_conf: {avg_conf:.2f}")

    # Cross-tabulation: What response types appear for each trigger type?
    print("\n" + "="*50)
    print("TRIGGER → RESPONSE CROSS-TAB (actual data)")
    print("="*50)

    cross_tab = {}
    for tr, rr in zip(trigger_results, response_results):
        if tr.label not in cross_tab:
            cross_tab[tr.label] = Counter()
        cross_tab[tr.label][rr.label] += 1

    for trigger_type in sorted(cross_tab.keys()):
        print(f"\n  {trigger_type}:")
        for resp_type, count in cross_tab[trigger_type].most_common(5):
            pct = count / sum(cross_tab[trigger_type].values()) * 100
            valid = "✓" if resp_type in TRIGGER_TO_VALID_RESPONSES.get(trigger_type, []) else " "
            print(f"    {valid} {resp_type:20} {count:4} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Build dialogue act classifiers")
    parser.add_argument("--build", action="store_true", help="Build both classifier indices")
    parser.add_argument("--validate", action="store_true", help="Run validation suite")
    parser.add_argument(
        "--test", action="store_true", help="Test on examples with neighbor inspection"
    )
    parser.add_argument("--label", action="store_true", help="Label our extracted data")
    parser.add_argument("--limit", type=int, default=1000, help="Limit for labeling")
    # Build options
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Don't balance classes (use weighted voting instead)",
    )
    parser.add_argument(
        "--no-mined",
        action="store_true",
        help="Don't include mined exemplars from clusters",
    )
    parser.add_argument(
        "--proportional",
        action="store_true",
        help="Use proportional sampling (ANSWER:2000, AGREE:400, etc.) instead of equal 500/class",
    )

    args = parser.parse_args()

    if args.build:
        cmd_build(
            balanced=not args.no_balance,
            include_mined=not args.no_mined,
            proportional=args.proportional,
        )

    if args.validate:
        cmd_validate()

    if args.test:
        cmd_test()

    if args.label:
        cmd_label(args.limit)

    if not any([args.build, args.validate, args.test, args.label]):
        parser.print_help()


if __name__ == "__main__":
    main()
