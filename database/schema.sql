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