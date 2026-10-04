-- Historical host schema v6 from coordinator commit b5fce34e496c61bb66edd53d93e8218306e95131
-- Includes the release-guard column installed by that initializer.

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    agent_alias TEXT,
    session_id TEXT,
    container_name TEXT,
    requested_count INTEGER NOT NULL,
    requested_devices TEXT,
    allow_external_busy INTEGER NOT NULL DEFAULT 0,
    not_before REAL NOT NULL,
    latest_start REAL NOT NULL,
    estimated_duration_seconds INTEGER NOT NULL,
    preemptible INTEGER NOT NULL DEFAULT 0,
    priority INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL,
    submitted_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    granted_devices TEXT,
    activation_deadline REAL,
    fence_token INTEGER,
    started_at REAL,
    expected_end REAL,
    heartbeat_at REAL,
    heartbeat_deadline REAL,
    pid INTEGER,
    message TEXT,
    requested_service_port INTEGER,
    service_port_choices TEXT,
    granted_service_port INTEGER
);
CREATE INDEX IF NOT EXISTS idx_tasks_queue
    ON tasks(state, priority DESC, submitted_at ASC);
CREATE TABLE IF NOT EXISTS holds (
    hold_id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    devices TEXT NOT NULL,
    not_before REAL NOT NULL,
    end_at REAL NOT NULL,
    reason TEXT,
    state TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_holds_window
    ON holds(state, not_before, end_at);
CREATE TABLE IF NOT EXISTS ports (
    port INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    task_id TEXT NOT NULL,
    owner TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ports_task
    ON ports(task_id);
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    at REAL NOT NULL,
    kind TEXT NOT NULL,
    task_id TEXT,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS message_peers (
    user TEXT NOT NULL,
    session_id TEXT NOT NULL,
    seen_at REAL NOT NULL,
    PRIMARY KEY(user, session_id)
);
CREATE TABLE IF NOT EXISTS messages (
    cursor INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL UNIQUE,
    sender TEXT NOT NULL,
    sender_session TEXT NOT NULL,
    recipient TEXT NOT NULL,
    recipient_session TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_recipient
    ON messages(recipient, recipient_session, cursor);

ALTER TABLE tasks ADD COLUMN process_guard TEXT;
CREATE TRIGGER IF NOT EXISTS protect_managed_process_release
    BEFORE UPDATE OF state ON tasks
    WHEN OLD.process_guard IS NOT NULL AND NEW.process_guard IS NOT NULL
         AND NEW.state IN ('released', 'cancelled', 'expired')
    BEGIN
        SELECT RAISE(ABORT, 'managed process guard must be verified before release');
    END;
