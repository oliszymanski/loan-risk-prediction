CREATE TABLE dim_customer {
    customer_id SERIAL PRIMARY KEY,
};

CREATE TABLE risk_segments {
    segment_id SERIAL PRIMARY KEY,
    segment_name VARCHAR(100) NOT NULL,
    pd_min DECIMAL(5,4) NOT NULL,
    pd_max DECIMAL(5,4) NOT NULL
    description TEXT;
};

CREATE TABLE risk_segments (
    segment_id SERIAL PRIMARY KEY,
    segment_name VARCHAR(50),
    pd_min DECIMAL(5,4),
    pd_max DECIMAL(5,4),
    description TEXT
);

INSERT INTO risk_segments (segment_name, pd_min, pd_max, description) VALUES
('Very Low Risk', 0.00, 0.10, 'Excellent credit quality'),
('Low Risk', 0.10, 0.25, 'Good credit quality'),
('Medium Risk', 0.25, 0.50, 'Moderate credit quality'),
('High Risk', 0.50, 0.75, 'Poor credit quality'),
('Very High Risk', 0.75, 1.00, 'Default imminent');