"""Default template definitions for TemplateMatcher - CLEANED UP VERSION.

This file contains only templates appropriate for casual iMessage replies.
- Removed: 21 assistant_query templates (wrong use case)
- Rewrote: 8 formal business templates to be casual
- Added: 10 new templates for common gaps

Total: ~70 templates (down from 91)
"""

from __future__ import annotations

from typing import Any


def get_minimal_fallback_templates() -> list[Any]:
    """Minimal templates for development when WS3 not available."""
    from models.templates import ResponseTemplate

    return [
        # ============================================================================
        # QUICK ACKNOWLEDGMENTS (5 templates)
        # ============================================================================
        ResponseTemplate(
            name="quick_ok",
            patterns=[
                "ok",
                "k",
                "kk",
                "okay",
                "okie",
                "okey",
                "alright",
                "alr",
            ],
            response="Got it!",
        ),
        ResponseTemplate(
            name="quick_affirmative",
            patterns=[
                "sure",
                "yep",
                "yup",
                "yeah",
                "ya",
                "ye",
                "yes",
                "definitely",
                "for sure",
                "fosho",
                "bet",
            ],
            response="Sounds good!",
        ),
        ResponseTemplate(
            name="quick_thanks",
            patterns=[
                "thx",
                "ty",
                "tysm",
                "thanks",
                "thank u",
                "thank you",
                "thanks!",
                "tyvm",
            ],
            response="np!",
        ),
        ResponseTemplate(
            name="quick_no_problem",
            patterns=[
                "np",
                "no prob",
                "no problem",
                "no worries",
                "nw",
                "all good",
                "nws",
            ],
            response="👍",
        ),
        ResponseTemplate(
            name="acknowledgment",
            patterns=[
                "Got it",
                "Understood",
                "Makes sense",
                "Sounds good",
                "Perfect",
                "Gotcha",
                "Say less",
            ],
            response="Great!",
        ),
        # ============================================================================
        # LOCATION & TIME (6 templates)
        # ============================================================================
        ResponseTemplate(
            name="on_my_way",
            patterns=[
                "omw",
                "on my way",
                "leaving now",
                "heading out",
                "be there in 10",
                "coming now",
                "just left",
                "headed over",
            ],
            response="See you soon!",
        ),
        ResponseTemplate(
            name="be_there_soon",
            patterns=[
                "be there soon",
                "almost there",
                "5 mins away",
                "pulling up",
                "around the corner",
                "just about there",
                "close",
            ],
            response="Great, see you in a bit!",
        ),
        ResponseTemplate(
            name="running_late",
            patterns=[
                "running late",
                "gonna be late",
                "running behind",
                "stuck in traffic",
                "be there in a bit",
                "sorry running late",
            ],
            response="No worries, take your time!",
        ),
        ResponseTemplate(
            name="where_are_you",
            patterns=[
                "where are you",
                "where r u",
                "where you at",
                "wya",
                "where u at",
                "you here yet",
                "are you close",
            ],
            response="On my way! Be there soon.",
        ),
        ResponseTemplate(
            name="what_time",
            patterns=[
                "what time",
                "when",
                "what time works",
                "when should we meet",
                "what time is good",
                "when are you free",
                "what time you thinking",
            ],
            response="Let me check! When works for you?",
        ),
        ResponseTemplate(
            name="time_proposal",
            patterns=[
                "how about 3",
                "does 5 work",
                "is 7 ok",
                "maybe around 2",
                "let's say noon",
                "how's 6pm",
                "works for me at 4",
                "that work?",
            ],
            response="That works!",
        ),
        # ============================================================================
        # SOCIAL PLANS (6 templates)
        # ============================================================================
        ResponseTemplate(
            name="hang_out_invite",
            patterns=[
                "wanna hang",
                "want to hang out",
                "down to hang",
                "u free",
                "you free",
                "wanna chill",
                "want to chill",
                "dtf",
                "what are you doing",
            ],
            response="Yeah, I'm down! What did you have in mind?",
        ),
        ResponseTemplate(
            name="dinner_plans",
            patterns=[
                "down for dinner",
                "wanna grab dinner",
                "want to get food",
                "hungry?",
                "wanna eat",
                "let's get food",
                "dinner tonight?",
                "want food?",
            ],
            response="I'm in! Where were you thinking?",
        ),
        ResponseTemplate(
            name="free_tonight",
            patterns=[
                "free tonight",
                "doing anything tonight",
                "busy tonight",
                "plans tonight",
                "what are you doing tonight",
                "got plans",
                "you doing anything",
            ],
            response="Let me check! What's up?",
        ),
        ResponseTemplate(
            name="coffee_drinks",
            patterns=[
                "let's grab coffee",
                "wanna get coffee",
                "coffee sometime",
                "grab a drink",
                "wanna get drinks",
                "drinks later",
                "get boba",
            ],
            response="Sounds great! When works for you?",
        ),
        ResponseTemplate(
            name="activity_invite",
            patterns=[
                "wanna go skiing",
                "down to ski",
                "want to watch",
                "gonna watch",
                "about to watch",
                "abt to watch",
                "wanna workout",
                "down to hike",
                "want to gym",
            ],
            response="I'm in! When are you thinking?",
        ),
        ResponseTemplate(
            name="agreement",
            patterns=[
                "that works too",
                "works for me",
                "that works",
                "works for me too",
                "sounds good to me",
                "i'm down",
                "down for that",
                "i'm in",
            ],
            response="Nice! What time?",
        ),
        # ============================================================================
        # REACTIONS & EXPRESSIONS (6 templates)
        # ============================================================================
        ResponseTemplate(
            name="laughter",
            patterns=[
                "lol",
                "lmao",
                "haha",
                "hahaha",
                "lolol",
                "rofl",
                "dying",
                "i'm dead",
            ],
            response="Haha right?!",
        ),
        ResponseTemplate(
            name="emoji_reaction",
            patterns=[
                "😂",
                "🤣",
                "😭",
                "😆",
                "🙃",
            ],
            response="😂",
        ),
        ResponseTemplate(
            name="positive_reaction",
            patterns=[
                "nice",
                "nice!",
                "awesome",
                "amazing",
                "love it",
                "so cool",
                "that's great",
                "dope",
                "sick",
                "fire",
            ],
            response="Right?!",
        ),
        ResponseTemplate(
            name="check_in",
            patterns=[
                "you there",
                "you there?",
                "u there",
                "hello?",
                "hey?",
                "you alive",
                "earth to you",
            ],
            response="I'm here! What's up?",
        ),
        ResponseTemplate(
            name="did_you_see",
            patterns=[
                "did you see my text",
                "did you get my message",
                "see my last message",
                "did you read that",
                "u see that",
                "did u see",
                "you see my message",
            ],
            response="Just saw it!",
        ),
        ResponseTemplate(
            name="appreciation",
            patterns=[
                "you're the best",
                "ur the best",
                "appreciate it",
                "thanks so much",
                "really appreciate it",
                "you rock",
                "ily",
                "love you",
            ],
            response="Aw, thanks! ❤️",
        ),
        # ============================================================================
        # FAREWELLS & CLOSINGS (5 templates)
        # ============================================================================
        ResponseTemplate(
            name="talk_later",
            patterns=[
                "ttyl",
                "talk later",
                "talk to you later",
                "catch you later",
                "later",
                "laters",
                "chat soon",
            ],
            response="Talk soon!",
        ),
        ResponseTemplate(
            name="goodnight",
            patterns=[
                "gn",
                "goodnight",
                "good night",
                "night",
                "nite",
                "sleep well",
                "sweet dreams",
            ],
            response="Goodnight! 😴",
        ),
        ResponseTemplate(
            name="goodbye",
            patterns=[
                "bye",
                "bye!",
                "cya",
                "see ya",
                "see you",
                "peace",
                "take care",
            ],
            response="Bye!",
        ),
        ResponseTemplate(
            name="brb",
            patterns=[
                "brb",
                "be right back",
                "one sec",
                "gimme a sec",
                "one minute",
                "give me a minute",
            ],
            response="No rush!",
        ),
        ResponseTemplate(
            name="question_response",
            patterns=[
                "idk",
                "i don't know",
                "dunno",
                "not sure",
                "no idea",
                "beats me",
            ],
            response="No worries!",
        ),
        # ============================================================================
        # NEGATION & REFUSAL (NEW - 2 templates)
        # ============================================================================
        ResponseTemplate(
            name="polite_decline",
            patterns=[
                "nah",
                "nope",
                "no thanks",
                "not really",
                "can't make it",
                "won't work for me",
                "pass",
            ],
            response="No worries!",
        ),
        ResponseTemplate(
            name="negation_ack",
            patterns=[
                "that doesn't work",
                "that won't work",
                "can't do that",
                "not gonna work",
            ],
            response="All good!",
        ),
        # ============================================================================
        # OPINIONS & AGREEMENT (NEW - 2 templates)
        # ============================================================================
        ResponseTemplate(
            name="opinion_ack",
            patterns=[
                "i think",
                "i believe",
                "in my opinion",
                "imo",
                "tbh",
                "honestly",
                "fr",
                "for real",
            ],
            response="Yeah, that makes sense",
        ),
        ResponseTemplate(
            name="same_agreement",
            patterns=[
                "same",
                "same here",
                "me too",
                "i agree",
                "totally",
                "exactly",
                "word",
            ],
            response="Fr fr",
        ),
        # ============================================================================
        # FLEXIBILITY & DELEGATION (NEW - 1 template)
        # ============================================================================
        ResponseTemplate(
            name="flexibility",
            patterns=[
                "i'm cool w anything",
                "i'm cool with anything",
                "whatever works",
                "up to you",
                "you decide",
                "i'm not picky",
                "either way",
                "fine with me",
                "whatever you want",
            ],
            response="Sounds good!",
        ),
        # ============================================================================
        # WAIT/PAUSE CONTEXTS (NEW - 1 template)
        # ============================================================================
        ResponseTemplate(
            name="wait_pause",
            patterns=[
                "wait",
                "hold up",
                "hold on",
                "wait a sec",
                "hang on",
                "one moment",
            ],
            response="What's up?",
        ),
        # ============================================================================
        # EXCLAMATIONS (NEW - 1 template)
        # ============================================================================
        ResponseTemplate(
            name="exclamation",
            patterns=[
                "tight",
                "fire",
                "damn",
                "dang",
                "no way",
                "wow",
            ],
            response="Right?!",
        ),
        # ============================================================================
        # STATUS UPDATES (NEW - 1 template)
        # ============================================================================
        ResponseTemplate(
            name="status_update",
            patterns=[
                "i'll be there",
                "heading out now",
                "on my way soon",
                "leaving in",
                "should be there",
                "i'm here",
                "just arrived",
            ],
            response="Cool!",
        ),
        # ============================================================================
        # MEETING & SCHEDULING (REWRITTEN - 4 templates)
        # Was formal, now casual
        # ============================================================================
        ResponseTemplate(
            name="meeting_confirmation",
            patterns=[
                "Confirming our meeting tomorrow",
                "Just confirming our call",
                "Confirming the meeting time",
                "See you at the meeting",
                "still on for",
                "we still meeting",
            ],
            response="Sounds good! See you then",
        ),
        ResponseTemplate(
            name="schedule_request",
            patterns=[
                "Can we schedule a meeting",
                "When are you free to meet",
                "Let's set up a call",
                "What times work for you",
                "Can we find a time to talk",
                "when can you meet",
            ],
            response="Sure! When works?",
        ),
        ResponseTemplate(
            name="file_receipt",
            patterns=[
                "I've attached the file",
                "Please find attached",
                "Here's the document",
                "Attached is the file you requested",
                "I'm sending over the file",
                "sent you the file",
                "here's the file",
            ],
            response="Got it! Thanks",
        ),
        ResponseTemplate(
            name="thank_you_ack",
            patterns=[
                "Thanks for sending the report",
                "Thank you for the update",
                "Thanks for letting me know",
                "Thank you for your email",
                "Thanks for the information",
                "thanks for the help",
                "thanks for that",
            ],
            response="Ofc!",
        ),
        # ============================================================================
        # OUT OF OFFICE (REWRITTEN - 1 template)
        # ============================================================================
        ResponseTemplate(
            name="out_of_office",
            patterns=[
                "I'll be out of office",
                "I'm on vacation",
                "I'll be unavailable",
                "Out of the office until",
                "Taking some time off",
                "going on vacation",
                "won't be around",
            ],
            response="Have fun!",
        ),
        # ============================================================================
        # FOLLOW UP (REWRITTEN - 1 template)
        # ============================================================================
        ResponseTemplate(
            name="follow_up",
            patterns=[
                "Just following up",
                "Wanted to check in",
                "Any updates on this",
                "Circling back on this",
                "Following up on my previous email",
                "did you get a chance",
            ],
            response="Thanks for the reminder!",
        ),
        # ============================================================================
        # APOLOGY (REWRITTEN - 1 template)
        # ============================================================================
        ResponseTemplate(
            name="apology",
            patterns=[
                "Sorry for the delay",
                "Apologies for the late response",
                "Sorry I missed your message",
                "My apologies for not responding sooner",
                "Sorry for the wait",
                "sorry i'm late",
            ],
            response="No worries at all!",
        ),
        # ============================================================================
        # GREETING (REWRITTEN - 1 template)
        # ============================================================================
        ResponseTemplate(
            name="greeting",
            patterns=[
                "Hi, how are you",
                "Hello, hope you're doing well",
                "Good morning",
                "Hey, hope all is well",
                "Hi there",
                "how's it going",
                "what's up",
                "how are you",
            ],
            response="Hey! I'm good, you?",
        ),
        # ============================================================================
        # GROUP CHAT TEMPLATES (31 templates)
        # Keep all - they're well-designed for group contexts
        # ============================================================================
        # --- Event Planning ---
        ResponseTemplate(
            name="group_event_when_works",
            patterns=[
                "when works for everyone",
                "what time works for everyone",
                "when is everyone free",
                "when are you all available",
                "what works for the group",
                "when can everyone make it",
                "what day works for all",
            ],
            response="I'm flexible! What times are you all thinking?",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_event_day_proposal",
            patterns=[
                "I can do Saturday",
                "Saturday works for me",
                "I'm free on Sunday",
                "Friday works",
                "I can make it on",
                "that day works for me",
                "I'm available then",
            ],
            response="That works for me too!",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_event_conflict",
            patterns=[
                "that doesn't work for me",
                "I can't do that day",
                "I have a conflict",
                "that time doesn't work",
                "I'm busy then",
                "can we do a different day",
                "any other options",
            ],
            response="No worries! What other times work for you?",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_event_locked_in",
            patterns=[
                "let's do Saturday then",
                "Saturday it is",
                "let's lock that in",
                "sounds like a plan",
                "we're all set then",
                "it's a date",
                "perfect, see everyone then",
            ],
            response="Sounds good! See everyone there!",
            is_group_template=True,
        ),
        # --- RSVP Coordination ---
        ResponseTemplate(
            name="group_rsvp_yes",
            patterns=[
                "count me in",
                "I'm in",
                "I'll be there",
                "yes I'm coming",
                "definitely coming",
                "I'm down",
                "sign me up",
                "add me to the list",
            ],
            response="Awesome, see you there!",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_rsvp_plus_one",
            patterns=[
                "I'll be there +1",
                "count me plus one",
                "I'm bringing someone",
                "can I bring a friend",
                "I'll be there with my partner",
                "plus one for me",
                "bringing my +1",
            ],
            response="Great, the more the merrier!",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_rsvp_no",
            patterns=[
                "can't make it",
                "I won't be able to come",
                "count me out",
                "I have to skip this one",
                "I can't come",
                "unfortunately I can't make it",
                "sorry I can't be there",
            ],
            response="No worries, we'll miss you! Maybe next time.",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_rsvp_maybe",
            patterns=[
                "I might be able to come",
                "I'll try to make it",
                "tentative yes",
                "put me down as a maybe",
                "I'll let you know",
                "not sure yet",
                "I'll confirm later",
            ],
            response="Sounds good, just let us know when you can!",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_rsvp_headcount",
            patterns=[
                "who's coming",
                "how many people so far",
                "what's the headcount",
                "who's confirmed",
                "how many are coming",
                "who all is going",
                "what's the count",
            ],
            response="Let me check - I think we have a few confirmed so far!",
            is_group_template=True,
            min_group_size=3,
        ),
        # --- Poll Responses ---
        ResponseTemplate(
            name="group_poll_vote_a",
            patterns=[
                "I vote for option A",
                "option A",
                "I prefer A",
                "A for me",
                "going with A",
                "my vote is A",
                "definitely A",
            ],
            response="Got it, A it is for me too!",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_poll_vote_b",
            patterns=[
                "I vote for option B",
                "option B",
                "I prefer B",
                "B for me",
                "going with B",
                "my vote is B",
                "definitely B",
            ],
            response="B sounds good!",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_poll_either",
            patterns=[
                "either works for me",
                "I'm fine with both",
                "no preference",
                "both options work",
                "I can go either way",
                "happy with whatever",
                "any option is fine",
            ],
            response="Same here, flexible on this one!",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_poll_create",
            patterns=[
                "let's do a poll",
                "let's vote on it",
                "should we vote",
                "let's put it to a vote",
                "what does everyone think",
                "can we get everyone's input",
                "let's see what everyone wants",
            ],
            response="Good idea! What are the options?",
            is_group_template=True,
            min_group_size=3,
        ),
        # --- Group Logistics ---
        ResponseTemplate(
            name="group_logistics_who_bringing",
            patterns=[
                "who's bringing what",
                "what should I bring",
                "who's bringing food",
                "should I bring anything",
                "what do we need",
                "who's handling what",
                "what's everyone bringing",
            ],
            response="I can bring drinks! What else do we need?",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_logistics_ill_handle",
            patterns=[
                "I'll handle the reservation",
                "I'll book it",
                "I can make the reservation",
                "I'll take care of it",
                "leave it to me",
                "I got this",
                "I'll set it up",
            ],
            response="You're the best! Thanks for handling that!",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_logistics_location",
            patterns=[
                "where are we meeting",
                "what's the address",
                "where should we go",
                "any suggestions for a place",
                "where's the spot",
                "what venue",
                "location suggestions",
            ],
            response="Good question! Anyone have ideas?",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_logistics_carpooling",
            patterns=[
                "anyone need a ride",
                "I can drive",
                "can someone pick me up",
                "let's carpool",
                "who's driving",
                "I need a ride",
                "anyone driving from downtown",
            ],
            response="I might need a ride! Where are you coming from?",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_logistics_splitting_bill",
            patterns=[
                "let's split the bill",
                "how should we split it",
                "I'll venmo everyone",
                "everyone pay their share",
                "let's split evenly",
                "how much do I owe",
                "I'll send the payment request",
            ],
            response="Sounds fair! Just let me know the amount.",
            is_group_template=True,
        ),
        # --- Celebratory Messages ---
        ResponseTemplate(
            name="group_celebration_birthday",
            patterns=[
                "happy birthday",
                "happy bday",
                "hbd",
                "hope you have a great birthday",
                "wishing you a happy birthday",
                "birthday wishes",
                "have an amazing birthday",
            ],
            response="Happy birthday! Hope it's amazing! 🎉",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_celebration_congrats",
            patterns=[
                "congrats everyone",
                "congratulations to all",
                "way to go team",
                "we did it",
                "great job everyone",
                "congrats all around",
                "proud of everyone",
            ],
            response="Congrats all! We crushed it! 🎉",
            is_group_template=True,
            min_group_size=3,
        ),
        ResponseTemplate(
            name="group_celebration_individual",
            patterns=[
                "congrats",
                "congratulations",
                "so proud of you",
                "well done",
                "amazing job",
                "you did it",
                "so happy for you",
            ],
            response="Congrats! That's awesome! 🎉",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_celebration_holiday",
            patterns=[
                "happy holidays",
                "happy new year",
                "merry christmas",
                "happy thanksgiving",
                "happy easter",
                "have a great holiday",
                "enjoy the holidays",
            ],
            response="Happy holidays to everyone! 🎊",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_celebration_thanks",
            patterns=[
                "thanks everyone",
                "thank you all",
                "appreciate everyone",
                "thanks to all of you",
                "grateful for this group",
                "thanks for everything",
                "you all are the best",
            ],
            response="Aw, this group is the best! ❤️",
            is_group_template=True,
            min_group_size=3,
        ),
        # --- Information Sharing ---
        ResponseTemplate(
            name="group_info_fyi",
            patterns=[
                "fyi",
                "for your information",
                "just so everyone knows",
                "heads up",
                "just a heads up",
                "wanted to let you all know",
                "quick update",
            ],
            response="Thanks for the heads up!",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_info_sharing",
            patterns=[
                "sharing with the group",
                "thought you'd all want to see this",
                "check this out everyone",
                "sharing this with everyone",
                "wanted to share this",
                "look what I found",
                "you all need to see this",
            ],
            response="Thanks for sharing! This is great!",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_info_update",
            patterns=[
                "update for everyone",
                "quick update for the group",
                "here's what's happening",
                "status update",
                "letting everyone know",
                "keeping everyone posted",
                "just an update",
            ],
            response="Thanks for the update! Good to know.",
            is_group_template=True,
        ),
        ResponseTemplate(
            name="group_info_reminder",
            patterns=[
                "reminder for everyone",
                "don't forget",
                "just a reminder",
                "quick reminder",
                "remember that",
                "reminding everyone",
                "friendly reminder",
            ],
            response="Thanks for the reminder!",
            is_group_template=True,
        ),
        # --- Large Group Specific (10+ people) ---
        ResponseTemplate(
            name="group_large_lost_track",
            patterns=[
                "sorry catching up on messages",
                "catching up on the chat",
                "so many messages",
                "what did I miss",
                "can someone summarize",
                "tldr of the chat",
                "filling in on missed messages",
            ],
            response="No worries! Here's the quick version...",
            is_group_template=True,
            min_group_size=10,
        ),
        ResponseTemplate(
            name="group_large_quiet_down",
            patterns=[
                "so many notifications",
                "my phone is blowing up",
                "this chat is active",
                "loving the energy",
                "lots of messages",
                "active chat today",
                "the group is popping",
            ],
            response="Haha right? Love this group's energy!",
            is_group_template=True,
            min_group_size=10,
        ),
        # --- Small Group Specific (3-5 people) ---
        ResponseTemplate(
            name="group_small_intimate",
            patterns=[
                "just us three",
                "just the four of us",
                "our little group",
                "the gang",
                "the crew",
                "the squad",
                "just us",
            ],
            response="Love our little crew! 💯",
            is_group_template=True,
            min_group_size=3,
            max_group_size=5,
        ),
    ]
