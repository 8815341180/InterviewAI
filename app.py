import streamlit as st
import re
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="InterviewAI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 InterviewAI")
st.subheader("AI-Powered Interview Practice & Evaluation Platform")


# =========================================================
# SKILLS
# =========================================================

ALL_SKILLS = [
    "Python", "Java", "C++", "C", "SQL", "MySQL", "MongoDB",
    "OOP", "Data Structures", "Exception Handling",
    "Machine Learning", "Deep Learning", "NLP",
    "Pandas", "NumPy", "Scikit-learn", "TensorFlow",
    "HTML", "CSS", "JavaScript", "React",
    "Django", "Flask", "Git", "GitHub",
    "Streamlit", "Power BI", "Excel"
]


# =========================================================
# QUESTION BANK
# =========================================================

question_bank = {

    "Python Developer": {

        "Easy": [
            {
                "question": "What is Python?",
                "answer": (
                    "Python is a high-level, interpreted, general-purpose "
                    "programming language. It is easy to learn and is widely "
                    "used for web development, data science, machine learning "
                    "and automation."
                ),
                "concepts": [
                    "High-level language",
                    "Interpreted language",
                    "General-purpose language",
                    "Easy to learn"
                ]
            },

            {
                "question": "What are variables in Python?",
                "answer": (
                    "A variable in Python is a name used to store a value. "
                    "The value of a variable can be changed during program "
                    "execution. For example, age = 20."
                ),
                "concepts": [
                    "Variable stores a value",
                    "Variable name",
                    "Value can change",
                    "Example"
                ]
            },

            {
                "question": "What is the difference between a list and a tuple?",
                "answer": (
                    "A list is mutable, which means its elements can be "
                    "changed after creation. A tuple is immutable, so its "
                    "elements cannot be changed after creation."
                ),
                "concepts": [
                    "List",
                    "Tuple",
                    "Mutable",
                    "Immutable"
                ]
            },

            {
                "question": "What are Python data types?",
                "answer": (
                    "Python provides different data types such as int, float, "
                    "string, boolean, list, tuple, set and dictionary."
                ),
                "concepts": [
                    "int",
                    "float",
                    "string",
                    "boolean",
                    "list",
                    "tuple",
                    "set",
                    "dictionary"
                ]
            }
        ],

        "Medium": [
            {
                "question": "Explain list comprehension in Python.",
                "answer": (
                    "List comprehension is a concise way to create a new list "
                    "using an expression and an optional condition. For example, "
                    "[x*x for x in range(5)] creates a list of squares."
                ),
                "concepts": [
                    "List comprehension",
                    "Expression",
                    "Condition",
                    "Example"
                ]
            },

            {
                "question": "What is exception handling in Python?",
                "answer": (
                    "Exception handling is used to handle runtime errors "
                    "without stopping the entire program. Python uses try, "
                    "except, else and finally blocks for exception handling."
                ),
                "concepts": [
                    "Runtime error",
                    "try",
                    "except",
                    "finally"
                ]
            },

            {
                "question": "Explain the concept of OOP in Python.",
                "answer": (
                    "OOP stands for Object-Oriented Programming. It is a "
                    "programming approach based on classes and objects. "
                    "The main concepts include encapsulation, inheritance, "
                    "polymorphism and abstraction."
                ),
                "concepts": [
                    "OOP",
                    "Class",
                    "Object",
                    "Encapsulation",
                    "Inheritance",
                    "Polymorphism",
                    "Abstraction"
                ]
            },

            {
                "question": "What is the difference between a class and an object?",
                "answer": (
                    "A class is a blueprint or template used to create objects. "
                    "An object is an instance of a class that contains data "
                    "and behavior."
                ),
                "concepts": [
                    "Class",
                    "Blueprint",
                    "Object",
                    "Instance"
                ]
            }
        ],

        "Hard": [
            {
                "question": "Explain decorators in Python with an example.",
                "answer": (
                    "A decorator is a function that modifies or extends the "
                    "behavior of another function without changing its source "
                    "code. Decorators are commonly written using the @ syntax."
                ),
                "concepts": [
                    "Decorator",
                    "Function",
                    "Modify behavior",
                    "@ syntax"
                ]
            },

            {
                "question": "What are generators in Python and why are they useful?",
                "answer": (
                    "Generators are functions that produce values one at a time "
                    "using the yield keyword. They are memory efficient because "
                    "they do not store all values in memory at once."
                ),
                "concepts": [
                    "Generator",
                    "yield",
                    "One value at a time",
                    "Memory efficient"
                ]
            },

            {
                "question": "Explain shallow copy and deep copy in Python.",
                "answer": (
                    "A shallow copy creates a new object but keeps references "
                    "to nested objects. A deep copy creates a completely "
                    "independent copy including nested objects."
                ),
                "concepts": [
                    "Shallow copy",
                    "Deep copy",
                    "Nested objects",
                    "References"
                ]
            },

            {
                "question": "How does Python memory management work?",
                "answer": (
                    "Python manages memory automatically using a private heap, "
                    "reference counting and garbage collection. Objects are "
                    "created in memory and unused objects can be removed "
                    "automatically."
                ),
                "concepts": [
                    "Memory management",
                    "Private heap",
                    "Reference counting",
                    "Garbage collection"
                ]
            }
        ]
    },


    "Java Developer": {

        "Easy": [
            {
                "question": "What is Java?",
                "answer": (
                    "Java is a high-level, object-oriented programming language "
                    "designed to be platform independent. Java programs run "
                    "through the Java Virtual Machine."
                ),
                "concepts": [
                    "High-level",
                    "Object-oriented",
                    "Platform independent",
                    "JVM"
                ]
            },

            {
                "question": "What is JVM?",
                "answer": (
                    "JVM stands for Java Virtual Machine. It executes Java "
                    "bytecode and allows Java programs to run on different "
                    "platforms."
                ),
                "concepts": [
                    "Java Virtual Machine",
                    "Bytecode",
                    "Execution",
                    "Platform independent"
                ]
            },

            {
                "question": "What is the difference between JDK and JRE?",
                "answer": (
                    "JDK is the Java Development Kit used to develop Java "
                    "applications. JRE is the Java Runtime Environment used "
                    "to run Java applications."
                ),
                "concepts": [
                    "JDK",
                    "JRE",
                    "Development",
                    "Runtime"
                ]
            },

            {
                "question": "What are variables in Java?",
                "answer": (
                    "A variable in Java is a named memory location used to "
                    "store data. Every variable has a data type and a value."
                ),
                "concepts": [
                    "Variable",
                    "Memory location",
                    "Data type",
                    "Value"
                ]
            }
        ],

        "Medium": [
            {
                "question": "Explain OOP concepts in Java.",
                "answer": (
                    "The main OOP concepts in Java are encapsulation, "
                    "inheritance, polymorphism and abstraction. These concepts "
                    "help organize programs using classes and objects."
                ),
                "concepts": [
                    "Encapsulation",
                    "Inheritance",
                    "Polymorphism",
                    "Abstraction"
                ]
            },

            {
                "question": "What is inheritance in Java?",
                "answer": (
                    "Inheritance is an OOP feature in which one class acquires "
                    "properties and methods of another class. It promotes "
                    "code reusability."
                ),
                "concepts": [
                    "Inheritance",
                    "Parent class",
                    "Child class",
                    "Code reusability"
                ]
            },

            {
                "question": "What is method overloading and overriding?",
                "answer": (
                    "Method overloading means having multiple methods with the "
                    "same name but different parameters in the same class. "
                    "Method overriding occurs when a child class provides its "
                    "own implementation of a parent class method."
                ),
                "concepts": [
                    "Overloading",
                    "Different parameters",
                    "Overriding",
                    "Child class"
                ]
            },

            {
                "question": "Explain exception handling in Java.",
                "answer": (
                    "Java exception handling is used to handle runtime errors. "
                    "It uses try, catch, finally, throw and throws keywords."
                ),
                "concepts": [
                    "Exception",
                    "try",
                    "catch",
                    "finally",
                    "throw",
                    "throws"
                ]
            }
        ],

        "Hard": [
            {
                "question": "Explain the internal working of HashMap in Java.",
                "answer": (
                    "HashMap stores key-value pairs using hashing. A hash "
                    "function determines the bucket where an entry is stored. "
                    "Collisions are handled using linked structures or trees "
                    "depending on the situation."
                ),
                "concepts": [
                    "HashMap",
                    "Key-value",
                    "Hashing",
                    "Bucket",
                    "Collision"
                ]
            },

            {
                "question": "What is multithreading in Java?",
                "answer": (
                    "Multithreading is the execution of multiple threads "
                    "concurrently within a program. It helps perform multiple "
                    "tasks efficiently."
                ),
                "concepts": [
                    "Multithreading",
                    "Thread",
                    "Concurrent execution",
                    "Multiple tasks"
                ]
            },

            {
                "question": "Explain garbage collection in Java.",
                "answer": (
                    "Garbage collection is an automatic memory management "
                    "process in Java. The JVM identifies objects that are no "
                    "longer reachable and removes them from memory."
                ),
                "concepts": [
                    "Garbage collection",
                    "JVM",
                    "Memory management",
                    "Unused objects"
                ]
            },

            {
                "question": "What is the difference between abstract class and interface?",
                "answer": (
                    "An abstract class can contain abstract and concrete "
                    "methods and can have instance variables. An interface "
                    "mainly defines a contract that implementing classes follow."
                ),
                "concepts": [
                    "Abstract class",
                    "Interface",
                    "Abstract methods",
                    "Concrete methods"
                ]
            }
        ]
    },


    "Data Scientist": {

        "Easy": [
            {
                "question": "What is Machine Learning?",
                "answer": (
                    "Machine Learning is a branch of artificial intelligence "
                    "that allows computers to learn patterns from data and "
                    "make predictions or decisions without being explicitly "
                    "programmed for every task."
                ),
                "concepts": [
                    "Machine Learning",
                    "Artificial Intelligence",
                    "Data",
                    "Patterns",
                    "Prediction"
                ]
            },

            {
                "question": "What is supervised learning?",
                "answer": (
                    "Supervised learning is a type of machine learning where "
                    "a model learns from labeled training data to predict "
                    "outputs for new data."
                ),
                "concepts": [
                    "Supervised learning",
                    "Labeled data",
                    "Training data",
                    "Prediction"
                ]
            },

            {
                "question": "What is a dataset?",
                "answer": (
                    "A dataset is a collection of related data used for "
                    "analysis, training machine learning models or testing."
                ),
                "concepts": [
                    "Dataset",
                    "Collection of data",
                    "Training",
                    "Testing"
                ]
            },

            {
                "question": "What is the difference between training and testing data?",
                "answer": (
                    "Training data is used to train a machine learning model. "
                    "Testing data is used to evaluate how well the trained "
                    "model performs on unseen data."
                ),
                "concepts": [
                    "Training data",
                    "Testing data",
                    "Model training",
                    "Evaluation"
                ]
            }
        ],

        "Medium": [
            {
                "question": "What is overfitting and how can you prevent it?",
                "answer": (
                    "Overfitting occurs when a model learns the training data "
                    "too closely and performs poorly on unseen data. It can "
                    "be reduced using cross-validation, regularization, "
                    "simpler models or more training data."
                ),
                "concepts": [
                    "Overfitting",
                    "Unseen data",
                    "Cross-validation",
                    "Regularization"
                ]
            },

            {
                "question": "Explain the difference between classification and regression.",
                "answer": (
                    "Classification predicts discrete categories such as spam "
                    "or not spam. Regression predicts continuous numerical "
                    "values such as house price."
                ),
                "concepts": [
                    "Classification",
                    "Regression",
                    "Categories",
                    "Continuous values"
                ]
            },

            {
                "question": "What is cross-validation?",
                "answer": (
                    "Cross-validation is a model evaluation technique in which "
                    "data is divided into multiple parts or folds. The model "
                    "is trained and evaluated multiple times using different "
                    "folds."
                ),
                "concepts": [
                    "Cross-validation",
                    "Folds",
                    "Training",
                    "Evaluation"
                ]
            },

            {
                "question": "Explain precision, recall and F1-score.",
                "answer": (
                    "Precision measures how many predicted positive cases are "
                    "actually positive. Recall measures how many actual positive "
                    "cases were correctly identified. F1-score is the harmonic "
                    "mean of precision and recall."
                ),
                "concepts": [
                    "Precision",
                    "Recall",
                    "F1-score",
                    "Harmonic mean"
                ]
            }
        ],

        "Hard": [
            {
                "question": "Explain how Random Forest works internally.",
                "answer": (
                    "Random Forest is an ensemble learning algorithm that "
                    "combines multiple decision trees. Each tree is trained "
                    "using random samples and features, and their predictions "
                    "are combined to produce the final result."
                ),
                "concepts": [
                    "Random Forest",
                    "Ensemble",
                    "Decision trees",
                    "Random samples",
                    "Random features"
                ]
            },

            {
                "question": "How would you handle an imbalanced dataset?",
                "answer": (
                    "An imbalanced dataset can be handled using techniques "
                    "such as oversampling, undersampling, SMOTE, class weights "
                    "and suitable evaluation metrics such as precision, recall "
                    "and F1-score."
                ),
                "concepts": [
                    "Imbalanced dataset",
                    "Oversampling",
                    "Undersampling",
                    "SMOTE",
                    "Class weights"
                ]
            },

            {
                "question": "Explain gradient descent mathematically.",
                "answer": (
                    "Gradient descent is an optimization algorithm used to "
                    "minimize a loss function. Model parameters are updated "
                    "in the opposite direction of the gradient using a "
                    "learning rate."
                ),
                "concepts": [
                    "Gradient descent",
                    "Loss function",
                    "Gradient",
                    "Learning rate",
                    "Parameters"
                ]
            },

            {
                "question": "How would you select important features for a machine learning model?",
                "answer": (
                    "Feature selection can be performed using correlation "
                    "analysis, statistical tests, feature importance, "
                    "recursive feature elimination and regularization."
                ),
                "concepts": [
                    "Feature selection",
                    "Correlation",
                    "Feature importance",
                    "RFE",
                    "Regularization"
                ]
            }
        ]
    },


    "Web Developer": {

        "Easy": [
            {
                "question": "What is HTML?",
                "answer": (
                    "HTML stands for HyperText Markup Language. It is used "
                    "to create the structure and content of web pages."
                ),
                "concepts": [
                    "HTML",
                    "HyperText Markup Language",
                    "Web page structure"
                ]
            },

            {
                "question": "What is CSS?",
                "answer": (
                    "CSS stands for Cascading Style Sheets. It is used to "
                    "style and design HTML elements on web pages."
                ),
                "concepts": [
                    "CSS",
                    "Cascading Style Sheets",
                    "Styling",
                    "Design"
                ]
            },

            {
                "question": "What is JavaScript?",
                "answer": (
                    "JavaScript is a programming language commonly used to "
                    "make web pages interactive and dynamic."
                ),
                "concepts": [
                    "JavaScript",
                    "Programming language",
                    "Interactive",
                    "Dynamic"
                ]
            },

            {
                "question": "What is the difference between frontend and backend?",
                "answer": (
                    "Frontend is the part of a web application that users "
                    "see and interact with. Backend handles server-side "
                    "logic, databases and APIs."
                ),
                "concepts": [
                    "Frontend",
                    "Backend",
                    "User interface",
                    "Server-side",
                    "Database"
                ]
            }
        ],

        "Medium": [
            {
                "question": "Explain the CSS box model.",
                "answer": (
                    "The CSS box model describes how an element is represented "
                    "as content, padding, border and margin."
                ),
                "concepts": [
                    "Box model",
                    "Content",
                    "Padding",
                    "Border",
                    "Margin"
                ]
            },

            {
                "question": "What is DOM in JavaScript?",
                "answer": (
                    "DOM stands for Document Object Model. It represents an "
                    "HTML document as a tree of objects that JavaScript can "
                    "access and modify."
                ),
                "concepts": [
                    "DOM",
                    "Document Object Model",
                    "HTML",
                    "Tree",
                    "JavaScript"
                ]
            },

            {
                "question": "What is REST API?",
                "answer": (
                    "A REST API is an interface that allows applications to "
                    "communicate over HTTP using methods such as GET, POST, "
                    "PUT and DELETE."
                ),
                "concepts": [
                    "REST API",
                    "HTTP",
                    "GET",
                    "POST",
                    "PUT",
                    "DELETE"
                ]
            },

            {
                "question": "Explain responsive web design.",
                "answer": (
                    "Responsive web design is an approach in which a website "
                    "automatically adjusts its layout and elements to work "
                    "properly on different screen sizes and devices."
                ),
                "concepts": [
                    "Responsive design",
                    "Layout",
                    "Screen size",
                    "Devices"
                ]
            }
        ],

        "Hard": [
            {
                "question": "Explain how a REST API works internally.",
                "answer": (
                    "A client sends an HTTP request to a server endpoint. "
                    "The server processes the request, performs required "
                    "operations and sends an HTTP response containing data "
                    "or an appropriate status code."
                ),
                "concepts": [
                    "Client",
                    "HTTP request",
                    "Server",
                    "Endpoint",
                    "HTTP response"
                ]
            },

            {
                "question": "What is event delegation in JavaScript?",
                "answer": (
                    "Event delegation is a technique where an event listener "
                    "is attached to a parent element instead of individual "
                    "child elements. Event bubbling allows the parent to "
                    "handle events from its children."
                ),
                "concepts": [
                    "Event delegation",
                    "Event listener",
                    "Parent",
                    "Event bubbling"
                ]
            },

            {
                "question": "Explain authentication and authorization in web applications.",
                "answer": (
                    "Authentication verifies who a user is, while authorization "
                    "determines what that authenticated user is allowed to do."
                ),
                "concepts": [
                    "Authentication",
                    "Authorization",
                    "User identity",
                    "Permissions"
                ]
            },

            {
                "question": "How would you optimize a slow web application?",
                "answer": (
                    "A slow web application can be optimized by reducing "
                    "unnecessary requests, optimizing database queries, "
                    "compressing assets, using caching and improving backend "
                    "performance."
                ),
                "concepts": [
                    "Performance",
                    "Database optimization",
                    "Caching",
                    "Compression",
                    "Backend"
                ]
            }
        ]
    }
}


# =========================================================
# HR QUESTIONS
# =========================================================

hr_questions = [
    {
        "question": "Tell me about yourself.",
        "answer": (
            "I am a B.Tech CSE student with an interest in software "
            "development and artificial intelligence. I have been learning "
            "Python, Java, SQL and machine learning, and I have worked on "
            "projects to apply these skills."
        ),
        "concepts": [
            "Education",
            "Technical skills",
            "Projects",
            "Career interest"
        ]
    },

    {
        "question": "What are your strengths?",
        "answer": (
            "My strengths include willingness to learn, problem-solving, "
            "consistency and the ability to learn new technologies."
        ),
        "concepts": [
            "Learning ability",
            "Problem-solving",
            "Consistency"
        ]
    },

    {
        "question": "Why should we hire you?",
        "answer": (
            "You should hire me because I have a strong willingness to learn, "
            "a technical foundation and practical project experience. "
            "I am ready to improve my skills and contribute to the team."
        ),
        "concepts": [
            "Technical skills",
            "Project experience",
            "Learning attitude",
            "Contribution"
        ]
    },

    {
        "question": "Where do you see yourself in five years?",
        "answer": (
            "In five years, I want to become a skilled software professional "
            "with strong technical and problem-solving abilities and take "
            "responsibility for challenging projects."
        ),
        "concepts": [
            "Career growth",
            "Technical skills",
            "Problem-solving",
            "Responsibility"
        ]
    }
]


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_text_from_pdf(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# =========================================================
# SKILL EXTRACTION
# =========================================================

def extract_skills(text):

    text_lower = text.lower()

    found_skills = []

    sorted_skills = sorted(
        ALL_SKILLS,
        key=len,
        reverse=True
    )

    for skill in sorted_skills:

        pattern = r"\b" + re.escape(
            skill.lower()
        ) + r"\b"

        if re.search(
            pattern,
            text_lower
        ):
            found_skills.append(skill)

    return found_skills


def extract_skills_from_jd(job_description):

    return extract_skills(
        job_description
    )


# =========================================================
# RESUME VS JD MATCH
# =========================================================

def calculate_resume_jd_match(
    resume_skills,
    required_skills
):

    if not required_skills:

        return 0, [], []

    matched = []

    missing = []

    resume_lower = [
        skill.lower()
        for skill in resume_skills
    ]

    for skill in required_skills:

        if skill.lower() in resume_lower:

            matched.append(skill)

        else:

            missing.append(skill)

    match_percentage = (
        len(matched)
        / len(required_skills)
    ) * 100

    return (
        round(match_percentage, 2),
        matched,
        missing
    )


# =========================================================
# ROADMAP
# =========================================================

def generate_roadmap(
    missing_skills
):

    roadmap = {

        "Python": [
            "Python basics",
            "Functions",
            "Lists and dictionaries",
            "OOP",
            "Exception handling",
            "Practice coding problems"
        ],

        "Java": [
            "Java basics",
            "OOP",
            "Collections",
            "Exception handling",
            "Multithreading",
            "DSA practice"
        ],

        "SQL": [
            "SELECT queries",
            "WHERE and ORDER BY",
            "GROUP BY",
            "JOINs",
            "Subqueries",
            "Practice SQL problems"
        ],

        "MySQL": [
            "Database basics",
            "Tables",
            "Primary and foreign keys",
            "JOINs",
            "Indexes",
            "Queries"
        ],

        "OOP": [
            "Class and object",
            "Encapsulation",
            "Inheritance",
            "Polymorphism",
            "Abstraction"
        ],

        "Data Structures": [
            "Arrays",
            "Strings",
            "Linked List",
            "Stack",
            "Queue",
            "Trees",
            "Graphs"
        ],

        "Machine Learning": [
            "Supervised learning",
            "Unsupervised learning",
            "Regression",
            "Classification",
            "Model evaluation",
            "Projects"
        ],

        "Deep Learning": [
            "Neural networks",
            "Activation functions",
            "Backpropagation",
            "CNN",
            "RNN"
        ],

        "NLP": [
            "Text preprocessing",
            "Tokenization",
            "TF-IDF",
            "Word embeddings",
            "Text classification"
        ],

        "Git": [
            "git init",
            "git add",
            "git commit",
            "git push",
            "Branching"
        ],

        "GitHub": [
            "Repositories",
            "README",
            "Branches",
            "Pull requests",
            "Project portfolio"
        ],

        "HTML": [
            "HTML structure",
            "Forms",
            "Tables",
            "Semantic HTML"
        ],

        "CSS": [
            "Selectors",
            "Box model",
            "Flexbox",
            "Grid",
            "Responsive design"
        ],

        "JavaScript": [
            "Variables",
            "Functions",
            "DOM",
            "Events",
            "Async JavaScript"
        ],

        "React": [
            "Components",
            "Props",
            "State",
            "Hooks",
            "API integration"
        ],

        "Django": [
            "Django basics",
            "Models",
            "Views",
            "Templates",
            "REST API"
        ],

        "Flask": [
            "Flask basics",
            "Routes",
            "Templates",
            "APIs",
            "Deployment"
        ]
    }

    return {
        skill: roadmap[skill]
        for skill in missing_skills
        if skill in roadmap
    }


# =========================================================
# ANSWER EVALUATION
# =========================================================

def evaluate_answer(
    answer,
    question_data
):

    question = question_data["question"]

    example_answer = question_data["answer"]

    concepts = question_data["concepts"]


    if not answer.strip():

        return {
            "score": 0,
            "similarity": 0,
            "keyword_score": 0,
            "matched_keywords": [],
            "missed_concepts": concepts,
            "feedback": "Please provide an answer."
        }


    # -----------------------------------------------------
    # TF-IDF SIMILARITY
    # -----------------------------------------------------

    try:

        vectorizer = TfidfVectorizer()

        vectors = vectorizer.fit_transform(
            [
                example_answer,
                answer
            ]
        )

        similarity = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0] * 100

    except:

        similarity = 0


    # -----------------------------------------------------
    # CONCEPT MATCHING
    # -----------------------------------------------------

    answer_lower = answer.lower()

    matched_concepts = []

    missed_concepts = []

    for concept in concepts:

        if concept.lower() in answer_lower:

            matched_concepts.append(
                concept
            )

        else:

            missed_concepts.append(
                concept
            )


    if concepts:

        keyword_score = (
            len(matched_concepts)
            / len(concepts)
        ) * 100

    else:

        keyword_score = 0


    # -----------------------------------------------------
    # COMPLETENESS
    # -----------------------------------------------------

    word_count = len(
        answer.split()
    )

    if word_count < 10:

        completeness = 40

    elif word_count < 25:

        completeness = 70

    else:

        completeness = 90


    # -----------------------------------------------------
    # FINAL SCORE
    # -----------------------------------------------------

    final_score = (
        similarity * 0.5
        + keyword_score * 0.3
        + completeness * 0.2
    )

    score = min(
        10,
        round(
            final_score / 10,
            1
        )
    )


    # -----------------------------------------------------
    # FEEDBACK
    # -----------------------------------------------------

    if score >= 8:

        feedback = (
            "Excellent answer. You covered the important "
            "concepts clearly."
        )

    elif score >= 6:

        feedback = (
            "Good answer, but you can improve it by adding "
            "more technical details and an example."
        )

    else:

        feedback = (
            "Your answer needs improvement. Review the "
            "example answer and focus on the missing concepts."
        )


    return {

        "score": score,

        "similarity": round(
            similarity,
            2
        ),

        "keyword_score": round(
            keyword_score,
            2
        ),

        "matched_keywords":
            matched_concepts,

        "missed_concepts":
            missed_concepts,

        "feedback":
            feedback
    }


# =========================================================
# ADAPTIVE DIFFICULTY
# =========================================================

def get_next_difficulty(score):

    if score >= 8:

        return "Hard"

    elif score >= 6:

        return "Medium"

    else:

        return "Easy"


# =========================================================
# GET ADAPTIVE QUESTION
# =========================================================

def get_adaptive_question(
    job_role,
    difficulty,
    used_questions
):

    role_questions = question_bank.get(
        job_role,
        {}
    )

    available_questions = (
        role_questions.get(
            difficulty,
            []
        )
    )

    for question_data in available_questions:

        if (
            question_data["question"]
            not in used_questions
        ):

            return question_data

    # If current difficulty questions are finished,
    # search other levels.

    for level in [
        "Easy",
        "Medium",
        "Hard"
    ]:

        for question_data in (
            role_questions.get(
                level,
                []
            )
        ):

            if (
                question_data["question"]
                not in used_questions
            ):

                return question_data

    return None


# =========================================================
# SESSION STATE
# =========================================================

if "questions" not in st.session_state:
    st.session_state.questions = []

if "current_question" not in st.session_state:
    st.session_state.current_question = 0

if "evaluations" not in st.session_state:
    st.session_state.evaluations = []

if "evaluated" not in st.session_state:
    st.session_state.evaluated = False

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

if "resume_skills" not in st.session_state:
    st.session_state.resume_skills = []

if "required_skills" not in st.session_state:
    st.session_state.required_skills = []

if "current_difficulty" not in st.session_state:
    st.session_state.current_difficulty = "Easy"

if "used_questions" not in st.session_state:
    st.session_state.used_questions = []

if "last_score" not in st.session_state:
    st.session_state.last_score = None


# =========================================================
# SIDEBAR - RESUME
# =========================================================

st.sidebar.header("📄 Resume")

uploaded_resume = st.sidebar.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

if uploaded_resume:

    if st.sidebar.button(
        "Extract Resume"
    ):

        resume_text = (
            extract_text_from_pdf(
                uploaded_resume
            )
        )

        st.session_state.resume_text = (
            resume_text
        )

        st.session_state.resume_skills = (
            extract_skills(
                resume_text
            )
        )

        st.sidebar.success(
            "Resume extracted successfully!"
        )


# =========================================================
# RESUME ANALYSIS
# =========================================================

if st.session_state.resume_text:

    st.header(
        "📄 Resume Analysis"
    )

    skills = (
        st.session_state.resume_skills
    )

    if skills:

        st.write(
            "### Detected Skills"
        )

        st.write(
            ", ".join(skills)
        )

    else:

        st.warning(
            "No known skills detected."
        )

    with st.expander(
        "View Resume Text"
    ):

        st.text(
            st.session_state.resume_text
        )


# =========================================================
# INTERVIEW SETUP
# =========================================================

st.header(
    "⚙️ Interview Setup"
)

col1, col2 = st.columns(2)

with col1:

    job_role = st.selectbox(
        "Select Job Role",
        [
            "Python Developer",
            "Java Developer",
            "Data Scientist",
            "Web Developer"
        ]
    )

    experience = st.selectbox(
        "Experience Level",
        [
            "Fresher",
            "0-2 Years",
            "2-5 Years",
            "5+ Years"
        ]
    )


with col2:

    skills_input = st.text_input(
        "Your Skills",
        placeholder="Python, SQL, Machine Learning"
    )

    interview_type = st.selectbox(
        "Interview Type",
        [
            "Technical",
            "HR",
            "Mixed"
        ]
    )


# =========================================================
# JOB DESCRIPTION
# =========================================================

st.header(
    "📋 Job Description"
)

job_description = st.text_area(
    "Paste Job Description",
    height=180,
    placeholder="Paste the company job description here..."
)


# =========================================================
# JD ANALYSIS
# =========================================================

required_skills = []

if job_description.strip():

    required_skills = (
        extract_skills_from_jd(
            job_description
        )
    )

    st.session_state.required_skills = (
        required_skills
    )

    if required_skills:

        st.success(
            "Detected JD Skills: "
            + ", ".join(
                required_skills
            )
        )

    else:

        st.info(
            "No known skills detected from this JD."
        )


# =========================================================
# RESUME VS JD
# =========================================================

if (
    st.session_state.resume_skills
    and required_skills
):

    st.header(
        "📊 Resume vs Job Description"
    )

    (
        match_percentage,
        matched,
        missing
    ) = calculate_resume_jd_match(
        st.session_state.resume_skills,
        required_skills
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Resume-JD Match",
        f"{match_percentage}%"
    )

    col2.metric(
        "Matching Skills",
        len(matched)
    )

    col3.metric(
        "Missing Skills",
        len(missing)
    )

    st.write(
        "### ✅ Matching Skills"
    )

    if matched:

        st.write(
            ", ".join(matched)
        )

    else:

        st.write(
            "No matching skills found."
        )

    st.write(
        "### ❌ Missing Skills"
    )

    if missing:

        st.write(
            ", ".join(missing)
        )

    else:

        st.success(
            "No major missing skills detected!"
        )


    # -----------------------------------------------------
    # ROADMAP
    # -----------------------------------------------------

    if missing:

        st.header(
            "🛣️ Personalized Preparation Roadmap"
        )

        roadmap = generate_roadmap(
            missing
        )

        for skill, topics in roadmap.items():

            with st.expander(
                f"📚 {skill}"
            ):

                for topic in topics:

                    st.write(
                        f"• {topic}"
                    )


# =========================================================
# START INTERVIEW
# =========================================================

if st.button(
    "🚀 Start Interview",
    type="primary"
):

    st.session_state.questions = []

    st.session_state.current_question = 0

    st.session_state.evaluations = []

    st.session_state.evaluated = False

    st.session_state.used_questions = []

    st.session_state.last_score = None

    st.session_state.current_difficulty = "Easy"


    if interview_type == "HR":

        st.session_state.questions = (
            hr_questions.copy()
        )


    else:

        first_question = (
            get_adaptive_question(
                job_role,
                "Easy",
                []
            )
        )

        if first_question:

            st.session_state.questions = [
                first_question
            ]


    st.success(
        "Interview started!"
    )


# =========================================================
# INTERVIEW
# =========================================================

if st.session_state.questions:

    st.header(
        "🎤 Interview"
    )

    current_index = (
        st.session_state.current_question
    )

    if (
        current_index
        < len(
            st.session_state.questions
        )
    ):

        question_data = (
            st.session_state.questions[
                current_index
            ]
        )

        question = (
            question_data["question"]
        )


        # -------------------------------------------------
        # QUESTION
        # -------------------------------------------------

        st.write(
            f"### Question {current_index + 1}"
        )

        st.info(
            question
        )

        st.caption(
            "Current Difficulty: "
            + st.session_state.current_difficulty
        )


        # -------------------------------------------------
        # ANSWER
        # -------------------------------------------------

        answer = st.text_area(
            "Your Answer",
            key=f"answer_{current_index}",
            height=180
        )


        # -------------------------------------------------
        # EVALUATE
        # -------------------------------------------------

        if st.button(
            "🧠 Evaluate Answer",
            key=f"evaluate_{current_index}"
        ):

            if not answer.strip():

                st.warning(
                    "Please enter your answer first."
                )

            else:

                result = evaluate_answer(
                    answer,
                    question_data
                )

                st.session_state.evaluations.append(
                    result
                )

                st.session_state.used_questions.append(
                    question
                )

                st.session_state.last_score = (
                    result["score"]
                )

                st.session_state.evaluated = True


                # =========================================
                # SCORE
                # =========================================

                st.subheader(
                    "📊 Evaluation Result"
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Score",
                    f"{result['score']}/10"
                )

                col2.metric(
                    "Similarity",
                    f"{result['similarity']}%"
                )

                col3.metric(
                    "Keyword Match",
                    f"{result['keyword_score']}%"
                )


                # =========================================
                # MATCHED CONCEPTS
                # =========================================

                if result[
                    "matched_keywords"
                ]:

                    st.write(
                        "### ✅ Matched Concepts"
                    )

                    st.write(
                        ", ".join(
                            result[
                                "matched_keywords"
                            ]
                        )
                    )


                # =========================================
                # FEEDBACK
                # =========================================

                st.write(
                    "### 💡 Feedback"
                )

                st.write(
                    result["feedback"]
                )


                # =========================================
                # WEAK ANSWER SUPPORT
                # =========================================

                if result["score"] < 8:

                    st.divider()

                    st.subheader(
                        "📚 Improve Your Answer"
                    )

                    # -------------------------------------
                    # EXAMPLE ANSWER
                    # -------------------------------------

                    st.write(
                        "### ✅ Example Answer"
                    )

                    st.info(
                        question_data["answer"]
                    )


                    # -------------------------------------
                    # WHAT WAS MISSED
                    # -------------------------------------

                    if result[
                        "missed_concepts"
                    ]:

                        st.write(
                            "### 🔍 What You Missed"
                        )

                        for concept in (
                            result[
                                "missed_concepts"
                            ]
                        ):

                            st.write(
                                f"• {concept}"
                            )


                    # -------------------------------------
                    # HOW TO IMPROVE
                    # -------------------------------------

                    st.write(
                        "### 💡 How to Improve"
                    )

                    if result["score"] < 5:

                        st.write(
                            "Your answer needs more "
                            "fundamental concepts. First "
                            "understand the definition, "
                            "then explain the important "
                            "points and give an example."
                        )

                    else:

                        st.write(
                            "Your basic answer is correct, "
                            "but add more important concepts "
                            "and a practical example to make "
                            "your answer stronger."
                        )


                else:

                    st.success(
                        "🎉 Strong answer! "
                        "You covered the important concepts."
                    )


                # =========================================
                # ADAPTIVE DIFFICULTY
                # =========================================

                next_difficulty = (
                    get_next_difficulty(
                        result["score"]
                    )
                )

                st.session_state.current_difficulty = (
                    next_difficulty
                )

                st.success(
                    "Next question difficulty: "
                    f"**{next_difficulty}**"
                )


        # -------------------------------------------------
        # NEXT QUESTION
        # -------------------------------------------------

        if st.session_state.evaluated:

            if st.button(
                "➡️ Next Question",
                key=f"next_{current_index}"
            ):

                next_difficulty = (
                    st.session_state.current_difficulty
                )


                if interview_type == "HR":

                    if (
                        current_index + 1
                        < len(
                            st.session_state.questions
                        )
                    ):

                        st.session_state.current_question += 1

                        st.session_state.evaluated = False

                        st.rerun()

                    else:

                        st.session_state.current_question += 1

                        st.session_state.evaluated = False

                        st.success(
                            "🎉 HR Interview Completed!"
                        )


                else:

                    next_question = (
                        get_adaptive_question(
                            job_role,
                            next_difficulty,
                            st.session_state.used_questions
                        )
                    )

                    if next_question:

                        st.session_state.questions.append(
                            next_question
                        )

                        st.session_state.current_question += 1

                        st.session_state.evaluated = False

                        st.rerun()

                    else:

                        st.session_state.current_question += 1

                        st.session_state.evaluated = False

                        st.success(
                            "🎉 Interview Completed!"
                        )


# =========================================================
# FINAL REPORT
# =========================================================

if (
    st.session_state.evaluations
    and st.session_state.current_question
    >= len(
        st.session_state.questions
    )
):

    st.header(
        "📊 Final Interview Report"
    )

    scores = [
        result["score"]
        for result in (
            st.session_state.evaluations
        )
    ]

    if scores:

        overall_score = round(
            sum(scores) / len(scores),
            2
        )

        st.metric(
            "Overall Interview Score",
            f"{overall_score}/10"
        )

        strong_answers = sum(
            score >= 8
            for score in scores
        )

        weak_answers = sum(
            score < 6
            for score in scores
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Strong Answers",
            strong_answers
        )

        col2.metric(
            "Answers Needing Improvement",
            weak_answers
        )


        st.write(
            "### 📈 Question-wise Performance"
        )

        for i, score in enumerate(
            scores
        ):

            st.write(
                f"Question {i + 1}: "
                f"**{score}/10**"
            )


        if st.session_state.required_skills:

            st.write(
                "### 🎯 JD Skills"
            )

            st.write(
                ", ".join(
                    st.session_state.required_skills
                )
            )


        if st.session_state.resume_skills:

            (
                match_percentage,
                matched,
                missing
            ) = calculate_resume_jd_match(
                st.session_state.resume_skills,
                st.session_state.required_skills
            )

            st.write(
                "### 📄 Resume-JD Match"
            )

            st.write(
                f"Match Score: "
                f"**{match_percentage}%**"
            )
        
# =========================================================
# INTERVIEW PERFORMANCE DASHBOARD
# =========================================================

st.divider()

st.header("📊 Interview Performance Dashboard")

if st.session_state.evaluations:

    scores = [
        result["score"]
        for result in st.session_state.evaluations
    ]

    total_questions = len(scores)

    average_score = round(
        sum(scores) / total_questions, 2
    )

    strong_answers = sum(
        score >= 8 for score in scores
    )

    weak_answers = sum(
        score < 6 for score in scores
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Questions Attempted",
        total_questions
    )

    col2.metric(
        "Average Score",
        f"{average_score}/10"
    )

    col3.metric(
        "Strong Answers",
        strong_answers
    )

    st.subheader("📈 Score Chart")

    chart_data = {
        "Question": [
            f"Q{i + 1}"
            for i in range(total_questions)
        ],
        "Score": scores
    }

    st.line_chart(
        chart_data,
        x="Question",
        y="Score"
    )

    st.subheader("🎯 Performance Summary")

    st.write(
        f"Answers needing improvement: {weak_answers}"
    )

    if average_score >= 8:
        st.success(
            "Excellent performance! Keep practicing."
        )
    elif average_score >= 6:
        st.info(
            "Good progress! Practice more to improve."
        )
    else:
        st.warning(
            "Keep practicing. Review your weak answers."
        )

else:
    st.info(
        "Complete at least one interview question "
        "to view your performance dashboard."
    )
