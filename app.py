from flask import Flask, request, jsonify, render_template
import sqlite3
import bcrypt
from jwt import encode, decode
import datetime
from functools import wraps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(os.path.abspath(os.path.dirname(__file__)), 'notes.db')}"

# Secret key for JWT encoding/decoding
app.config['SECRET_KEY'] = os.getenv('JWT_KEY')

# Function to connect to the database
def get_db_connection():
    conn = sqlite3.connect('notes.db')
    conn.row_factory = sqlite3.Row  # This will allow us to access rows as dictionaries
    return conn

# Initialize database tables
def initialize_database():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    
    # Create notes table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT,
            note TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
        
    conn.commit()
    conn.close()

# Token validation decorator
def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None

        # Check if the token is in the request header
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header[7:]
            else:
                token = auth_header

        if not token:
            return jsonify({'error': 'Token is missing!'}), 403

        try:
            # Decode the token
            data = decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = data['user_id']
        except Exception as e:
            return jsonify({'error': 'Token is invalid!'}), 403

        return f(current_user, *args, **kwargs)

    return decorated_function

# Register Route
@app.route('/register', methods=['POST'])
def register():
    # Get user details from the request body
    username = request.json.get('username')
    email = request.json.get('email')
    password = request.json.get('password')

    # Validate the input
    if not username or not email or not password:
        return jsonify({"error": "All fields (username, email, password) are required."}), 400

    # Hash the password using bcrypt
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # Insert the new user into the database
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('''
            INSERT INTO users (username, email, password_hash) 
            VALUES (?, ?, ?)
        ''', (username, email, hashed_password))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return jsonify({"message": "User registered successfully!", "user_id": user_id}), 201
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error": "Username or email already exists."}), 400

# Login Route
@app.route('/login', methods=['POST'])
def login():
    # Get user credentials from the request body
    username = request.json.get('username')
    password = request.json.get('password')

    if not username or not password:
        return jsonify({"error": "Username and password are required."}), 400

    # Check if the user exists in the database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()

    if user is None:
        return jsonify({"error": "Invalid username or password."}), 401

    # Check if the password matches the hashed password in the database
    if bcrypt.checkpw(password.encode('utf-8'), user['password_hash']):
        # Generate a JWT token
        token = encode({
            'user_id': user['id'],
            'username': user['username'],
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm='HS256')

        return jsonify({
            "message": "Login successful!", 
            "token": token, 
            "user_id": user['id'],
            "username": user['username']
        }), 200
    else:
        return jsonify({"error": "Invalid username or password."}), 401

# Add note route
@app.route('/add_note', methods=['POST'])
@token_required
def add_note(current_user):
    # Get note content from the request
    title = request.json.get('title', '')
    note = request.json.get('note')
    
    # Validate input
    if not note:
        return jsonify({"error": "Note content is required"}), 400
    
    # Insert the new note into the database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO notes (user_id, title, note) 
        VALUES (?, ?, ?)
    ''', (current_user, title, note))
    
    note_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return jsonify({
        "message": "Note added successfully!",
        "note_id": note_id
    }), 201

# Get all notes for a user
@app.route('/get_notes', methods=['GET'])
@token_required
def get_notes(current_user):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, title, note, created_at 
        FROM notes 
        WHERE user_id = ? 
        ORDER BY created_at DESC
    ''', (current_user,))
    
    notes = cursor.fetchall()
    conn.close()
    
    # Convert the notes to a list of dictionaries
    notes_list = []
    for note in notes:
        notes_list.append({
            "id": note['id'],
            "title": note['title'],
            "content": note['note'],
            "created_at": note['created_at']
        })
    
    return jsonify({"notes": notes_list})

# Get note graph for similarity analysis
@app.route('/get_note_graph', methods=['GET'])
@token_required
def get_note_graph(current_user):
    # Get optional similarity threshold from request
    threshold = request.args.get('threshold', 0.3, type=float)
    
    # Get notes for the user
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, title, note FROM notes WHERE user_id = ?', (current_user,))
    notes = cursor.fetchall()
    conn.close()
    
    if not notes:
        return jsonify({"error": "No notes found for this user"}), 404
    
    # Extract the note content for similarity comparison
    note_texts = [note['note'] for note in notes]
    note_ids = [note['id'] for note in notes]
    
    # Create TF-IDF vectorizer and compute the TF-IDF matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(note_texts)
    
    # Compute cosine similarity between all pairs of notes
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Create the graph
    G = nx.Graph()
    
    # Add nodes for each note
    for i, note in enumerate(notes):
        G.add_node(note['id'], 
                  id=note['id'], 
                  title=note['title'] or f"Note {note['id']}", 
                  content=note['note'][:100] + "..." if len(note['note']) > 100 else note['note'])
    
    # Add edges between notes if similarity is above the threshold
    for i in range(len(notes)):
        for j in range(i + 1, len(notes)):
            similarity_score = float(similarity_matrix[i][j])
            if similarity_score >= threshold:
                G.add_edge(note_ids[i], note_ids[j], weight=round(similarity_score, 2))
    
    # Prepare the graph data for the response
    nodes = [{"id": node, "title": G.nodes[node]["title"], "preview": G.nodes[node]["content"]} 
             for node in G.nodes]
    
    edges = [{"source": edge[0], 
              "target": edge[1], 
              "weight": G[edge[0]][edge[1]]['weight']} 
             for edge in G.edges]
    
    return jsonify({
        "nodes": nodes, 
        "edges": edges,
        "threshold": threshold
    })

# User profile route
@app.route('/profile', methods=['GET'])
@token_required
def profile(current_user):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, username, email FROM users WHERE id = ?', (current_user,))
    user = cursor.fetchone()
    
    # Count notes for the user
    cursor.execute('SELECT COUNT(*) FROM notes WHERE user_id = ?', (current_user,))
    note_count = cursor.fetchone()[0]
    
    conn.close()
    
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    return jsonify({
        "user_id": user['id'],
        "username": user['username'],
        "email": user['email'],
        "note_count": note_count
    })


@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")

@app.route("/dashboard", methods=['GET'])
def dashboard():
    return render_template("dashboard.html")



if __name__ == '__main__':
    # Only initialize the database if running locally (not in production)
    if app.env == 'development':
        initialize_database()
    app.run(debug=True)  # this line will be used only when running locally

# if __name__ == '__main__':
#     initialize_database()
#     app.run(debug=True)