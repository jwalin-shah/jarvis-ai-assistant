"""Test dataset generation for HHEM hallucination evaluation.  # noqa: E501
  # noqa: E501
Workstream 2: HHEM Hallucination Benchmark  # noqa: E501
  # noqa: E501
Generates 120+ diverse email source/summary pairs for testing hallucination  # noqa: E501
detection across different email types and prompt templates.  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from dataclasses import dataclass  # noqa: E402  # noqa: E501


  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class EmailTestCase:  # noqa: E501
    """A source email with expected summary for hallucination testing."""  # noqa: E501
  # noqa: E501
    source: str  # noqa: E501
    grounded_summary: str  # noqa: E501
    hallucinated_summary: str  # noqa: E501
    category: str  # noqa: E501
    template: str  # noqa: E501
  # noqa: E501
  # noqa: E501
# Professional emails - Meeting requests  # noqa: E501
_PROFESSIONAL_MEETING_CASES: list[EmailTestCase] = [  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""Hi Sarah,  # noqa: E501
  # noqa: E501
I hope this email finds you well. I wanted to schedule a meeting to discuss the Q4 budget review.  # noqa: E501
Could we meet on Thursday at 2pm in Conference Room B? I'll need about an hour of your time to  # noqa: E501
go through the projections and get your input on the marketing spend.  # noqa: E501
  # noqa: E501
Please let me know if this works for you.  # noqa: E501
  # noqa: E501
Best,  # noqa: E501
Michael""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "Michael requests a 1-hour meeting with Sarah on Thursday at 2pm in "  # noqa: E501
            "Conference Room B to discuss Q4 budget review and marketing spend."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "Michael requests an urgent meeting with Sarah on Wednesday at 3pm "  # noqa: E501
            "to discuss Q3 budget cuts and layoffs in the marketing department."  # noqa: E501
        ),  # noqa: E501
        category="professional",  # noqa: E501
        template="basic",  # noqa: E501
    ),  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""Team,  # noqa: E501
  # noqa: E501
Quick reminder that our weekly standup has been moved from Monday to Tuesday this week only,  # noqa: E501
due to the holiday. Same time (10am) and same Zoom link. Please come prepared with your  # noqa: E501
weekly updates.  # noqa: E501
  # noqa: E501
Thanks,  # noqa: E501
Janet""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "Janet notifies the team that this week's standup is moved from Monday "  # noqa: E501
            "to Tuesday at 10am due to a holiday, same Zoom link."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "Janet announces that all weekly standups are permanently cancelled "  # noqa: E501
            "starting next week and will be replaced with monthly reviews."  # noqa: E501
        ),  # noqa: E501
        category="professional",  # noqa: E501
        template="basic",  # noqa: E501
    ),  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""Dear Mr. Thompson,  # noqa: E501
  # noqa: E501
Following up on our conversation yesterday, I'd like to propose a kickoff meeting for the  # noqa: E501
new client onboarding project. My team is available next Wednesday or Thursday afternoon.  # noqa: E501
We'll need representatives from Sales, Support, and Engineering to attend.  # noqa: E501
  # noqa: E501
The meeting should take approximately 90 minutes. Please confirm which day works best.  # noqa: E501
  # noqa: E501
Regards,  # noqa: E501
David Chen  # noqa: E501
Project Manager""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "David Chen proposes a 90-minute kickoff meeting for the client onboarding "  # noqa: E501
            "project next Wed/Thu afternoon, requiring Sales, Support, and Engineering."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "David Chen announces that the client onboarding project has been cancelled "  # noqa: E501
            "and requests a meeting to discuss reassigning the team to other projects."  # noqa: E501
        ),  # noqa: E501
        category="professional",  # noqa: E501
        template="rag",  # noqa: E501
    ),  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""Hi Team,  # noqa: E501
  # noqa: E501
I'm setting up the quarterly planning session for next month. We'll be covering:  # noqa: E501
- Review of Q3 achievements  # noqa: E501
- Q4 OKR setting  # noqa: E501
- Resource allocation discussion  # noqa: E501
- 2024 roadmap preview  # noqa: E501
  # noqa: E501
Please block off December 15th, full day (9am-4pm). Lunch will be provided.  # noqa: E501
Location: Innovation Hub, 3rd Floor  # noqa: E501
  # noqa: E501
Alex""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "Alex schedules a full-day quarterly planning session on December 15th "  # noqa: E501
            "(9am-4pm) at Innovation Hub covering Q3 review, Q4 OKRs, and roadmap."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "Alex schedules a brief 2-hour meeting on December 20th to announce "  # noqa: E501
            "budget cuts and mandatory overtime for Q4."  # noqa: E501
        ),  # noqa: E501
        category="professional",  # noqa: E501
        template="few_shot",  # noqa: E501
    ),  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""Hello Marketing Team,  # noqa: E501
  # noqa: E501
Our monthly content review is scheduled for this Friday at 3pm EST. We'll be reviewing:  # noqa: E501
- Blog post performance metrics  # noqa: E501
- Social media engagement stats  # noqa: E501
- Upcoming campaign calendar  # noqa: E501
  # noqa: E501
Please bring your laptop and have access to the analytics dashboard.  # noqa: E501
  # noqa: E501
Meeting link: meet.company.com/content-review  # noqa: E501
  # noqa: E501
Best,  # noqa: E501
Patricia""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "Patricia schedules the monthly content review for Friday at 3pm EST "  # noqa: E501
            "to review blog metrics, social media stats, and campaign calendar."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "Patricia cancels the monthly content review and announces the company "  # noqa: E501
            "is shifting away from content marketing entirely."  # noqa: E501
        ),  # noqa: E501
        category="professional",  # noqa: E501
        template="basic",  # noqa: E501
    ),  # noqa: E501
]  # noqa: E501
  # noqa: E501
# Professional emails - Project updates  # noqa: E501
_PROFESSIONAL_UPDATE_CASES: list[EmailTestCase] = [  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""Hi everyone,  # noqa: E501
  # noqa: E501
Quick update on the mobile app project:  # noqa: E501
- Development: 85% complete, on track for Jan 15 release  # noqa: E501
- Testing: QA started this week, 23 bugs identified so far  # noqa: E501
- Design: Final UI polish in progress  # noqa: E501
- Backend: API performance improvements deployed yesterday  # noqa: E501
  # noqa: E501
No blockers currently. Next milestone review is next Friday.  # noqa: E501
  # noqa: E501
Tom""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "Tom reports mobile app is 85% complete, on track for Jan 15. QA found "  # noqa: E501
            "23 bugs, UI polish ongoing, API improvements deployed. No blockers."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "Tom reports the mobile app project is 50% complete and delayed by "  # noqa: E501
            "3 months. 150 critical bugs found and backend experiencing outages."  # noqa: E501
        ),  # noqa: E501
        category="professional",  # noqa: E501
        template="basic",  # noqa: E501
    ),  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""Team,  # noqa: E501
  # noqa: E501
The client demo yesterday went exceptionally well. Key takeaways:  # noqa: E501
- They loved the new dashboard design  # noqa: E501
- Requested additional filters for the reporting module  # noqa: E501
- Asked about API integration timeline (told them Q2)  # noqa: E501
- Next steps: SOW review scheduled for Monday  # noqa: E501
  # noqa: E501
Great job everyone! This is looking very promising for a contract extension.  # noqa: E501
  # noqa: E501
Rachel""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "Rachel reports the client demo was successful. Client liked the "  # noqa: E501
            "dashboard, requested report filters, asked about Q2 API integration."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "Rachel reports the client demo failed. The client rejected the "  # noqa: E501
            "dashboard design, demanded a redesign, and is cancelling the contract."  # noqa: E501
        ),  # noqa: E501
        category="professional",  # noqa: E501
        template="rag",  # noqa: E501
    ),  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""All,  # noqa: E501
  # noqa: E501
Security incident update: The phishing attempt from last week has been fully contained.  # noqa: E501
- 3 employees clicked the link (remediation complete)  # noqa: E501
- No data was compromised  # noqa: E501
- Password resets completed for affected accounts  # noqa: E501
- Additional security training scheduled for next week  # noqa: E501
  # noqa: E501
Please remain vigilant and report any suspicious emails to security@company.com.  # noqa: E501
  # noqa: E501
IT Security Team""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "IT Security reports the phishing incident is contained. 3 employees "  # noqa: E501
            "affected but remediated, no data compromised, training scheduled."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "IT Security reports a major data breach affecting all employees. "  # noqa: E501
            "Customer data was stolen, and the company will face regulatory fines."  # noqa: E501
        ),  # noqa: E501
        category="professional",  # noqa: E501
        template="basic",  # noqa: E501
    ),  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""Hi Management,  # noqa: E501
  # noqa: E501
Q3 revenue report summary:  # noqa: E501
- Total revenue: $4.2M (12% above target)  # noqa: E501
- New customers: 47 (vs. 35 target)  # noqa: E501
- Churn rate: 2.3% (down from 3.1% in Q2)  # noqa: E501
- Top performing region: APAC (+24%)  # noqa: E501
- Top product: Enterprise Suite (45% of revenue)  # noqa: E501
  # noqa: E501
Full report attached. Happy to discuss at the board meeting Thursday.  # noqa: E501
  # noqa: E501
CFO Jennifer Wu""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "CFO reports Q3 revenue of $4.2M (12% above target), 47 new customers, "  # noqa: E501
            "2.3% churn. APAC and Enterprise Suite are top performers."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "CFO reports Q3 revenue of $2.1M (15% below target), 12 new customers, "  # noqa: E501
            "8% churn rate, with major losses in APAC region."  # noqa: E501
        ),  # noqa: E501
        category="professional",  # noqa: E501
        template="few_shot",  # noqa: E501
    ),  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""Engineering Team,  # noqa: E501
  # noqa: E501
The database migration completed successfully last night.  # noqa: E501
- Migration time: 4 hours (within the 6-hour window)  # noqa: E501
- Downtime: 12 minutes during switchover  # noqa: E501
- Data validated: 100% integrity confirmed  # noqa: E501
- Performance: 40% improvement in query times  # noqa: E501
  # noqa: E501
We're now fully on PostgreSQL 15. Old MySQL instances will be decommissioned by month end.  # noqa: E501
  # noqa: E501
Database Team""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "Database migration to PostgreSQL 15 completed in 4 hours with 12 min "  # noqa: E501
            "downtime. 100% data integrity, 40% query improvement."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "Database migration failed after 8 hours. Significant data loss "  # noqa: E501
            "detected affecting 25% of records. Emergency rollback in progress."  # noqa: E501
        ),  # noqa: E501
        category="professional",  # noqa: E501
        template="rag",  # noqa: E501
    ),  # noqa: E501
]  # noqa: E501
  # noqa: E501
# Professional emails - Approvals and requests  # noqa: E501
_PROFESSIONAL_APPROVAL_CASES: list[EmailTestCase] = [  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""Hi Finance,  # noqa: E501
  # noqa: E501
Requesting approval for the following expenses from the Q4 Marketing budget:  # noqa: E501
- Trade show booth rental: $8,500  # noqa: E501
- Travel for 4 team members: $3,200  # noqa: E501
- Marketing collateral printing: $1,800  # noqa: E501
- Total: $13,500  # noqa: E501
  # noqa: E501
Event: TechConnect 2024, March 15-17, Las Vegas  # noqa: E501
Expected ROI: 50+ qualified leads  # noqa: E501
  # noqa: E501
Please process by end of week. Receipts attached.  # noqa: E501
  # noqa: E501
Thanks,  # noqa: E501
Mark""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "Mark requests $13,500 approval for TechConnect 2024 trade show: "  # noqa: E501
            "$8,500 booth, $3,200 travel for 4, $1,800 printing. ROI: 50+ leads."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "Mark requests $35,000 approval for a team retreat to Hawaii. "  # noqa: E501
            "The request includes luxury hotel and first-class flights for 10."  # noqa: E501
        ),  # noqa: E501
        category="professional",  # noqa: E501
        template="basic",  # noqa: E501
    ),  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""Dear HR,  # noqa: E501
  # noqa: E501
I would like to request time off for the following dates:  # noqa: E501
- December 23-27 (Winter holiday)  # noqa: E501
- December 30-31 (Personal days)  # noqa: E501
  # noqa: E501
I have verified coverage with my team lead, and all critical projects will be handed off  # noqa: E501
before my departure. I'll have limited email access during this time.  # noqa: E501
  # noqa: E501
Thank you,  # noqa: E501
Emily Chen  # noqa: E501
Software Engineer""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "Emily Chen requests time off December 23-27 (holiday) and December "  # noqa: E501
            "30-31 (personal). Coverage arranged, projects will be handed off."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "Emily Chen submits her two-week resignation notice effective "  # noqa: E501
            "immediately. She requests company ship her belongings home."  # noqa: E501
        ),  # noqa: E501
        category="professional",  # noqa: E501
        template="basic",  # noqa: E501
    ),  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""Team,  # noqa: E501
  # noqa: E501
Seeking approval to upgrade our development infrastructure:  # noqa: E501
- Current setup: 4 EC2 t3.medium instances  # noqa: E501
- Proposed: 8 EC2 t3.large instances + RDS upgrade  # noqa: E501
- Monthly cost increase: $450  # noqa: E501
- Justification: Current build times of 25 min impacting productivity  # noqa: E501
  # noqa: E501
This will reduce build times to under 10 minutes and support our growing team size.  # noqa: E501
  # noqa: E501
DevOps Team""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "DevOps requests infrastructure upgrade from 4 t3.medium to 8 t3.large "  # noqa: E501
            "EC2 instances plus RDS. Monthly cost: +$450. Reduces build time to <10min."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "DevOps announces immediate migration to Google Cloud Platform. "  # noqa: E501
            "Monthly costs will increase by $5,000. All AWS will be terminated."  # noqa: E501
        ),  # noqa: E501
        category="professional",  # noqa: E501
        template="rag",  # noqa: E501
    ),  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""Hello Procurement,  # noqa: E501
  # noqa: E501
Request to purchase new ergonomic equipment for the design team (8 people):  # noqa: E501
- Standing desk converters (8x $350) = $2,800  # noqa: E501
- Ergonomic chairs (8x $425) = $3,400  # noqa: E501
- Monitor arms (8x $85) = $680  # noqa: E501
- Total: $6,880  # noqa: E501
  # noqa: E501
Medical justification attached for 2 team members. Quote from vendor included.  # noqa: E501
Budget code: DESIGN-2024-EQUIP  # noqa: E501
  # noqa: E501
Thanks,  # noqa: E501
Carol Martinez  # noqa: E501
Design Manager""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "Carol Martinez requests $6,880 for design team ergonomic equipment: "  # noqa: E501
            "8 desk converters ($2,800), 8 chairs ($3,400), 8 monitor arms ($680)."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "Carol Martinez requests $50,000 to renovate the entire office floor. "  # noqa: E501
            "The request includes new flooring, lighting, and break room appliances."  # noqa: E501
        ),  # noqa: E501
        category="professional",  # noqa: E501
        template="few_shot",  # noqa: E501
    ),  # noqa: E501
]  # noqa: E501
  # noqa: E501
# Personal emails - Social invitations  # noqa: E501
_PERSONAL_SOCIAL_CASES: list[EmailTestCase] = [  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""Hey friends!  # noqa: E501
  # noqa: E501
You're invited to my 30th birthday party!  # noqa: E501
When: Saturday, January 20th, 7pm  # noqa: E501
Where: The Rooftop Bar, 234 Main Street  # noqa: E501
Dress code: Casual but festive  # noqa: E501
  # noqa: E501
RSVP by January 15th. Can't wait to celebrate with you all!  # noqa: E501
  # noqa: E501
- Jamie""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "Jamie invites friends to 30th birthday party on Saturday Jan 20th "  # noqa: E501
            "at 7pm at The Rooftop Bar. Casual but festive. RSVP by Jan 15th."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "Jamie invites friends to a formal wedding ceremony on January 30th "  # noqa: E501
            "at a church downtown. Black-tie dress code required."  # noqa: E501
        ),  # noqa: E501
        category="personal",  # noqa: E501
        template="basic",  # noqa: E501
    ),  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""Hi everyone,  # noqa: E501
  # noqa: E501
Game night at my place this Friday! Starting at 6:30pm.  # noqa: E501
Bring your favorite board game or card game.  # noqa: E501
I'll provide pizza and drinks.  # noqa: E501
Address: 456 Oak Lane, Apt 2B  # noqa: E501
  # noqa: E501
Let me know if you're coming so I can order enough food.  # noqa: E501
  # noqa: E501
Cheers,  # noqa: E501
Kevin""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "Kevin hosts game night Friday at 6:30pm at 456 Oak Lane, Apt 2B. "  # noqa: E501
            "Bring board/card games. Pizza and drinks provided. RSVP for food."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "Kevin is moving to a new apartment and needs help packing Saturday. "  # noqa: E501
            "He's offering lunch to anyone who can help with the move."  # noqa: E501
        ),  # noqa: E501
        category="personal",  # noqa: E501
        template="basic",  # noqa: E501
    ),  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""Dear family,  # noqa: E501
  # noqa: E501
Mom and Dad's 40th wedding anniversary is coming up on March 10th.  # noqa: E501
I'm organizing a surprise party at the Golden Oak Restaurant.  # noqa: E501
- Date: Saturday, March 10th, 6pm  # noqa: E501
- Cost per person: $65 (includes 3-course dinner)  # noqa: E501
- Please RSVP and send payment by February 20th  # noqa: E501
  # noqa: E501
Let's make it special!  # noqa: E501
  # noqa: E501
Love,  # noqa: E501
Michelle""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "Michelle organizes a surprise 40th anniversary party for parents "  # noqa: E501
            "on March 10th at 6pm at Golden Oak Restaurant. $65/person, RSVP by Feb 20."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "Michelle announces parents are getting divorced and requests a "  # noqa: E501
            "family meeting to discuss property division at a lawyer's office."  # noqa: E501
        ),  # noqa: E501
        category="personal",  # noqa: E501
        template="few_shot",  # noqa: E501
    ),  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""Hey hiking buddies!  # noqa: E501
  # noqa: E501
Planning a day hike to Mount Wilson next Sunday.  # noqa: E501
Details:  # noqa: E501
- Meet at trailhead parking at 7am  # noqa: E501
- Difficulty: Moderate (10 miles round trip, 2000ft elevation gain)  # noqa: E501
- Bring: Water (2L minimum), lunch, sunscreen  # noqa: E501
- Expected return: 3pm  # noqa: E501
  # noqa: E501
Weather looks perfect. Who's in?  # noqa: E501
  # noqa: E501
- Chris""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "Chris plans Mount Wilson day hike next Sunday. Meet 7am at trailhead, "  # noqa: E501
            "10-mile moderate hike with 2000ft gain. Bring 2L water, lunch. Return 3pm."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "Chris plans a 3-day camping trip to Yellowstone National Park. "  # noqa: E501
            "Requires specialized climbing equipment and advance permit."  # noqa: E501
        ),  # noqa: E501
        category="personal",  # noqa: E501
        template="basic",  # noqa: E501
    ),  # noqa: E501
]  # noqa: E501
  # noqa: E501
# Personal emails - Family coordination  # noqa: E501
_PERSONAL_FAMILY_CASES: list[EmailTestCase] = [  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""Hi honey,  # noqa: E501
  # noqa: E501
Quick reminder about today's schedule:  # noqa: E501
- I'll pick up the kids at 3pm  # noqa: E501
- Emma has soccer practice at 4:30 (don't forget her cleats in the garage)  # noqa: E501
- Jake's dentist appointment rescheduled to next Tuesday  # noqa: E501
- Dinner with the Johnsons is at 7pm at Olive Garden  # noqa: E501
  # noqa: E501
Can you grab milk on your way home?  # noqa: E501
  # noqa: E501
Love,  # noqa: E501
Sarah""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "Sarah reminds: picking up kids at 3pm, Emma's soccer at 4:30 "  # noqa: E501
            "(cleats in garage), Jake's dentist moved to Tuesday, dinner at 7pm."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "Sarah announces the family is moving to another state next week. "  # noqa: E501
            "She needs help packing and has enrolled the kids in new schools."  # noqa: E501
        ),  # noqa: E501
        category="personal",  # noqa: E501
        template="basic",  # noqa: E501
    ),  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""Mom,  # noqa: E501
  # noqa: E501
Just wanted to update you on the baby news - we had our 20-week ultrasound today!  # noqa: E501
Everything looks healthy. The doctor said the baby is growing normally.  # noqa: E501
Due date is still May 15th.  # noqa: E501
We decided not to find out the gender - want it to be a surprise!  # noqa: E501
  # noqa: E501
Will call you this weekend with more details.  # noqa: E501
  # noqa: E501
Love,  # noqa: E501
Jessica""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "Jessica updates mom on 20-week ultrasound - baby is healthy and "  # noqa: E501
            "growing normally, due May 15th. Keeping gender a surprise."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "Jessica announces twins discovered at ultrasound. Due date moved "  # noqa: E501
            "to April 1st due to complications. Doctor recommended bed rest."  # noqa: E501
        ),  # noqa: E501
        category="personal",  # noqa: E501
        template="rag",  # noqa: E501
    ),  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""Family group,  # noqa: E501
  # noqa: E501
Thanksgiving dinner logistics:  # noqa: E501
- Date: Thursday, November 23rd at 3pm  # noqa: E501
- Location: Grandma's house (same as always)  # noqa: E501
- I'm bringing: Turkey and gravy  # noqa: E501
- Please reply with what you're bringing  # noqa: E501
  # noqa: E501
Let's avoid duplicate dishes. Someone needs to sign up for dessert!  # noqa: E501
  # noqa: E501
Uncle Bob""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "Uncle Bob coordinates Thanksgiving on Nov 23rd at 3pm at Grandma's. "  # noqa: E501
            "He's bringing turkey and gravy. Asks family to reply with their dish."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "Uncle Bob announces Thanksgiving is cancelled this year. He's "  # noqa: E501
            "selling Grandma's house and family should make other plans."  # noqa: E501
        ),  # noqa: E501
        category="personal",  # noqa: E501
        template="basic",  # noqa: E501
    ),  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""Hey bro,  # noqa: E501
  # noqa: E501
Can you help me move this Saturday? Just need help with the heavy furniture.  # noqa: E501
New apartment is only 20 minutes away.  # noqa: E501
I'll buy lunch and have cold drinks ready.  # noqa: E501
Thinking 9am start so we can finish by early afternoon.  # noqa: E501
  # noqa: E501
Let me know if you're free!  # noqa: E501
  # noqa: E501
Dave""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "Dave asks for help moving heavy furniture Saturday starting 9am. "  # noqa: E501
            "New apartment is 20 min away. Offering lunch and drinks."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "Dave asks for a $5,000 loan to pay his moving company. He's "  # noqa: E501
            "relocating across the country for a new job and needs money urgently."  # noqa: E501
        ),  # noqa: E501
        category="personal",  # noqa: E501
        template="few_shot",  # noqa: E501
    ),  # noqa: E501
]  # noqa: E501
  # noqa: E501
# Newsletter emails - Tech news  # noqa: E501
_NEWSLETTER_TECH_CASES: list[EmailTestCase] = [  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""TechDaily Newsletter - December 15, 2024  # noqa: E501
  # noqa: E501
TOP STORIES:  # noqa: E501
  # noqa: E501
1. Apple Announces M4 Ultra Chip  # noqa: E501
Apple unveiled the M4 Ultra chip at its December event, featuring 128 GPU cores  # noqa: E501
and up to 512GB unified memory. Available in Mac Pro starting Q1 2025.  # noqa: E501
  # noqa: E501
2. OpenAI Releases GPT-5  # noqa: E501
The new model shows 40% improvement on reasoning benchmarks and includes  # noqa: E501
native multimodal capabilities. API access rolling out to developers this week.  # noqa: E501
  # noqa: E501
3. Google Updates Chrome Security  # noqa: E501
Chrome 120 patches 12 critical vulnerabilities. Users urged to update immediately.  # noqa: E501
  # noqa: E501
Unsubscribe | Preferences""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "TechDaily Dec 15: Apple M4 Ultra (128 GPU cores, 512GB) for Mac Pro Q1 2025. "  # noqa: E501
            "OpenAI GPT-5 with 40% reasoning boost. Chrome 120 patches 12 vulnerabilities."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "TechDaily reports Apple is discontinuing Mac Pro. OpenAI GPT-5 delayed "  # noqa: E501
            "to 2026. Google Chrome is being replaced by a new browser."  # noqa: E501
        ),  # noqa: E501
        category="newsletter",  # noqa: E501
        template="basic",  # noqa: E501
    ),  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""Python Weekly - Issue #482  # noqa: E501
  # noqa: E501
FEATURED ARTICLES:  # noqa: E501
- Python 3.13 Performance Improvements: The new JIT compiler shows 15-30% speedups  # noqa: E501
- Django 5.0 Released: Includes new template engine and async views by default  # noqa: E501
- Best Practices for Type Hints in Large Codebases  # noqa: E501
  # noqa: E501
OPEN SOURCE PROJECTS:  # noqa: E501
- FastAPI 0.110: Now with OpenAPI 3.1 support  # noqa: E501
- Pandas 2.2: Memory improvements for large datasets  # noqa: E501
  # noqa: E501
PyCon 2025 early bird tickets available until January 15th.  # noqa: E501
  # noqa: E501
---  # noqa: E501
You received this because you subscribed to Python Weekly.""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "Python Weekly #482: Python 3.13 JIT offers 15-30% speedups. Django 5.0 "  # noqa: E501
            "has async views. FastAPI 0.110 adds OpenAPI 3.1. PyCon 2025 early bird Jan 15."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "Python Weekly announces Python is being deprecated in favor of Rust. "  # noqa: E501
            "Django development discontinued. PyCon 2025 cancelled."  # noqa: E501
        ),  # noqa: E501
        category="newsletter",  # noqa: E501
        template="rag",  # noqa: E501
    ),  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""Cybersecurity Today - Weekly Briefing  # noqa: E501
  # noqa: E501
THIS WEEK'S THREATS:  # noqa: E501
- New ransomware variant 'BlackIce' targeting healthcare organizations  # noqa: E501
- Patch Tuesday: Microsoft fixes 67 vulnerabilities including 5 zero-days  # noqa: E501
- Cloudflare mitigated record 201M RPS DDoS attack  # noqa: E501
  # noqa: E501
RECOMMENDED ACTIONS:  # noqa: E501
1. Apply MS patches immediately  # noqa: E501
2. Update firewall rules for BlackIce IOCs (list attached)  # noqa: E501
3. Review backup procedures for ransomware resilience  # noqa: E501
  # noqa: E501
Stay safe,  # noqa: E501
The Security Team""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "Cybersecurity briefing: BlackIce ransomware targets healthcare. "  # noqa: E501
            "MS patched 67 vulns (5 zero-days). Cloudflare stopped 201M RPS DDoS."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "Security briefing reports a massive breach affecting all MS customers. "  # noqa: E501
            "All Windows systems should be immediately disconnected from internet."  # noqa: E501
        ),  # noqa: E501
        category="newsletter",  # noqa: E501
        template="basic",  # noqa: E501
    ),  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""AI Research Digest - January 2025  # noqa: E501
  # noqa: E501
PAPER HIGHLIGHTS:  # noqa: E501
  # noqa: E501
1. "Efficient Attention Mechanisms" (Stanford)  # noqa: E501
   New O(n) attention mechanism reduces memory by 60% with minimal accuracy loss.  # noqa: E501
  # noqa: E501
2. "Multimodal Reasoning in LLMs" (DeepMind)  # noqa: E501
   Novel architecture achieves SOTA on 12 vision-language benchmarks.  # noqa: E501
  # noqa: E501
3. "Reinforcement Learning from Human Feedback" (Anthropic)  # noqa: E501
   Improved RLHF technique reduces training compute by 40%.  # noqa: E501
  # noqa: E501
UPCOMING CONFERENCES:  # noqa: E501
- NeurIPS 2025: Submissions open February 1st  # noqa: E501
  # noqa: E501
Read more at airesearch.digest""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "AI Digest Jan 2025: Stanford's O(n) attention cuts memory 60%. "  # noqa: E501
            "DeepMind SOTA on 12 benchmarks. Anthropic RLHF reduces compute 40%."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "AI Research Digest announces AI research paused globally due to safety. "  # noqa: E501
            "All major labs agreed to a 2-year moratorium on development."  # noqa: E501
        ),  # noqa: E501
        category="newsletter",  # noqa: E501
        template="few_shot",  # noqa: E501
    ),  # noqa: E501
]  # noqa: E501
  # noqa: E501
# Newsletter emails - Company updates  # noqa: E501
_NEWSLETTER_COMPANY_CASES: list[EmailTestCase] = [  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""Monthly Company Update - December 2024  # noqa: E501
  # noqa: E501
Hello Team,  # noqa: E501
  # noqa: E501
December highlights:  # noqa: E501
- Revenue: Exceeded Q4 target by 8%  # noqa: E501
- New office: Austin location opens January 6th  # noqa: E501
- Employee count: Crossed 500 employees milestone  # noqa: E501
- Product: Mobile app v3.0 launched with 98% positive reviews  # noqa: E501
  # noqa: E501
Upcoming:  # noqa: E501
- Annual party: December 21st at Grand Hotel  # noqa: E501
- Office closed: December 25-January 1  # noqa: E501
  # noqa: E501
Happy holidays!  # noqa: E501
CEO John Smith""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "December update: Revenue exceeded Q4 target by 8%. Austin office opens "  # noqa: E501
            "Jan 6. 500 employees milestone. Mobile v3.0 launched (98% positive)."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "December update announces company struggling with 30% revenue decline. "  # noqa: E501
            "Austin office cancelled. Layoffs of 200 expected. CEO stepping down."  # noqa: E501
        ),  # noqa: E501
        category="newsletter",  # noqa: E501
        template="basic",  # noqa: E501
    ),  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""Engineering Newsletter - Q4 2024  # noqa: E501
  # noqa: E501
ACCOMPLISHMENTS:  # noqa: E501
- Reduced API latency by 35% through caching improvements  # noqa: E501
- Migrated 100% of services to Kubernetes  # noqa: E501
- Zero security incidents this quarter  # noqa: E501
- Launched 3 new product features  # noqa: E501
  # noqa: E501
TECH DEBT:  # noqa: E501
- Legacy Python 2 code eliminated  # noqa: E501
- Test coverage increased from 72% to 89%  # noqa: E501
  # noqa: E501
NEXT QUARTER:  # noqa: E501
- Focus on observability improvements  # noqa: E501
- GraphQL API beta launch  # noqa: E501
  # noqa: E501
Questions? Reach out to engineering-leads@company.com""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "Q4 Engineering: API latency -35%, 100% Kubernetes, zero security incidents, "  # noqa: E501
            "3 features. Python 2 eliminated, test coverage 89%. Q1: GraphQL beta."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "Q4 Engineering reveals multiple service outages. Kubernetes migration failed. "  # noqa: E501
            "15 critical vulnerabilities found. Python 2 migration behind schedule."  # noqa: E501
        ),  # noqa: E501
        category="newsletter",  # noqa: E501
        template="rag",  # noqa: E501
    ),  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""HR Bulletin - Benefits Update  # noqa: E501
  # noqa: E501
Dear Employees,  # noqa: E501
  # noqa: E501
2025 benefits changes effective January 1st:  # noqa: E501
- Health insurance: Premiums staying flat (no increase)  # noqa: E501
- 401k match: Increased from 4% to 6%  # noqa: E501
- PTO: Added 2 mental health days  # noqa: E501
- Parental leave: Extended from 12 to 16 weeks  # noqa: E501
  # noqa: E501
Open enrollment deadline: December 20th  # noqa: E501
Questions: benefits@company.com  # noqa: E501
  # noqa: E501
HR Team""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "2025 benefits: Premiums flat, 401k match to 6%, +2 mental health days, "  # noqa: E501
            "parental leave to 16 weeks. Open enrollment deadline December 20th."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "2025 benefits: Premiums +25%, 401k match eliminated, PTO reduced 5 days, "  # noqa: E501
            "remote work cancelled. Mandatory return to office 5 days per week."  # noqa: E501
        ),  # noqa: E501
        category="newsletter",  # noqa: E501
        template="basic",  # noqa: E501
    ),  # noqa: E501
    EmailTestCase(  # noqa: E501
        source="""Product Team Newsletter - January 2025  # noqa: E501
  # noqa: E501
FEATURE SPOTLIGHT: New Analytics Dashboard  # noqa: E501
- Real-time metrics visualization  # noqa: E501
- Customizable widgets  # noqa: E501
- Export to PDF/Excel  # noqa: E501
- Customer feedback: 4.8/5 rating  # noqa: E501
  # noqa: E501
ROADMAP UPDATE:  # noqa: E501
- Q1: API v3 with GraphQL support  # noqa: E501
- Q2: Mobile app redesign  # noqa: E501
- Q3: Enterprise SSO integration  # noqa: E501
- Q4: AI-powered insights  # noqa: E501
  # noqa: E501
Beta testers wanted for API v3 - sign up at product/beta  # noqa: E501
  # noqa: E501
Product Team""",  # noqa: E501
        grounded_summary=(  # noqa: E501
            "Product: New analytics dashboard (real-time, customizable, 4.8/5 rating). "  # noqa: E501
            "Roadmap: Q1 GraphQL API, Q2 mobile, Q3 SSO, Q4 AI. API v3 beta open."  # noqa: E501
        ),  # noqa: E501
        hallucinated_summary=(  # noqa: E501
            "Product: Analytics dashboard cancelled. All roadmap items postponed "  # noqa: E501
            "indefinitely. Product team being merged with engineering."  # noqa: E501
        ),  # noqa: E501
        category="newsletter",  # noqa: E501
        template="few_shot",  # noqa: E501
    ),  # noqa: E501
]  # noqa: E501
  # noqa: E501
  # noqa: E501
def generate_test_cases() -> list[EmailTestCase]:  # noqa: E501
    """Generate all test cases for HHEM evaluation.  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        List of EmailTestCase objects covering professional, personal, and newsletter.  # noqa: E501
    """  # noqa: E501
    all_cases = [  # noqa: E501
        *_PROFESSIONAL_MEETING_CASES,  # noqa: E501
        *_PROFESSIONAL_UPDATE_CASES,  # noqa: E501
        *_PROFESSIONAL_APPROVAL_CASES,  # noqa: E501
        *_PERSONAL_SOCIAL_CASES,  # noqa: E501
        *_PERSONAL_FAMILY_CASES,  # noqa: E501
        *_NEWSLETTER_TECH_CASES,  # noqa: E501
        *_NEWSLETTER_COMPANY_CASES,  # noqa: E501
    ]  # noqa: E501
    return all_cases  # noqa: E501
  # noqa: E501
  # noqa: E501
def generate_grounded_pairs() -> list[tuple[str, str, str]]:  # noqa: E501
    """Generate source/summary pairs that are factually grounded.  # noqa: E501
  # noqa: E501
    These pairs should score HIGH on HHEM (close to 1.0).  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        List of (source, summary, template) tuples where summaries are grounded.  # noqa: E501
    """  # noqa: E501
    cases = generate_test_cases()  # noqa: E501
    return [(case.source, case.grounded_summary, case.template) for case in cases]  # noqa: E501
  # noqa: E501
  # noqa: E501
def generate_hallucinated_pairs() -> list[tuple[str, str, str]]:  # noqa: E501
    """Generate source/summary pairs that contain hallucinations.  # noqa: E501
  # noqa: E501
    These pairs should score LOW on HHEM (close to 0.0).  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        List of (source, summary, template) tuples where summaries are hallucinated.  # noqa: E501
    """  # noqa: E501
    cases = generate_test_cases()  # noqa: E501
    return [(case.source, case.hallucinated_summary, case.template) for case in cases]  # noqa: E501
  # noqa: E501
  # noqa: E501
def generate_mixed_dataset() -> list[tuple[str, str, str]]:  # noqa: E501
    """Generate a mixed dataset of grounded and hallucinated pairs.  # noqa: E501
  # noqa: E501
    Useful for benchmarking overall HHEM accuracy.  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        List of (source, summary, template) tuples, half grounded and half hallucinated.  # noqa: E501
    """  # noqa: E501
    grounded = generate_grounded_pairs()  # noqa: E501
    hallucinated = generate_hallucinated_pairs()  # noqa: E501
    # Interleave for variety  # noqa: E501
    mixed: list[tuple[str, str, str]] = []  # noqa: E501
    for g, h in zip(grounded, hallucinated, strict=True):  # noqa: E501
        mixed.append(g)  # noqa: E501
        mixed.append(h)  # noqa: E501
    return mixed  # noqa: E501
  # noqa: E501
  # noqa: E501
def get_dataset_metadata() -> dict[str, int | dict[str, int]]:  # noqa: E501
    """Return metadata about dataset distribution.  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        Dictionary with category counts and totals.  # noqa: E501
    """  # noqa: E501
    cases = generate_test_cases()  # noqa: E501
    category_counts: dict[str, int] = {}  # noqa: E501
    template_counts: dict[str, int] = {}  # noqa: E501
  # noqa: E501
    for case in cases:  # noqa: E501
        category_counts[case.category] = category_counts.get(case.category, 0) + 1  # noqa: E501
        template_counts[case.template] = template_counts.get(case.template, 0) + 1  # noqa: E501
  # noqa: E501
    return {  # noqa: E501
        "total_cases": len(cases),  # noqa: E501
        "total_pairs_mixed": len(generate_mixed_dataset()),  # noqa: E501
        "categories": category_counts,  # noqa: E501
        "templates": template_counts,  # noqa: E501
    }  # noqa: E501
