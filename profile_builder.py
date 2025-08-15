class ProfileBuilder:
    def __init__(self, use_llm=True, llm=None, max_questions=6):
        self.responses = {}
        self.state = 0
        self.use_llm = use_llm
        self.llm = llm
        self.max_questions = max_questions
        self.last_question = ""

    def next_question(self):
        if self.state == 0:
            self.last_question =  "What is your name?"
            return self.last_question
        if self.use_llm and self.llm:
            return self.generate_dynamic_question()
        else:
            raise Exception("LLM is not enabled or not provided.")

    def generate_dynamic_question(self):
        profile_so_far = "\n".join(f"{k}: {v}" for k, v in self.responses.items())
        prompt = f"""
You are a helpful AI assistant collecting details to recommend an ideal job for a user.
So far, the responses are:
{profile_so_far if profile_so_far else 'None yet'}

Ask the next relevant question to better understand the user’s profile. Keep it short and polite. 
Return only the question, not any extra text or explanation.
"""
        response = self.llm(prompt).strip()
        self.last_question = response
        return response

    def receive_response(self, user_input):
        self.responses[self.last_question] = user_input
        self.state += 1

    def is_complete(self):
        return self.state >= self.max_questions

    def combined_text(self):
        return " ".join(self.responses.values())

    @property
    def name(self):
        for question, answer in self.responses.items():
            if "name" in question.lower():
                return answer.strip()
        return "Anonymous"