CREATE DATABASE IF NOT EXISTS healthcare_capstone;
USE healthcare_capstone;

CREATE TABLE healthcare_providers (
    national_provider_identifier BIGINT,
    provider_name VARCHAR(255),
    provider_first_name VARCHAR(100),
    provider_middle_initial VARCHAR(10),
    credentials VARCHAR(100),
    gender VARCHAR(20),
    entity_type VARCHAR(100),
    provider_type VARCHAR(150),
    street_address VARCHAR(255),
    city VARCHAR(100),
    zip_code VARCHAR(20),
    state_code VARCHAR(10),
    country_code VARCHAR(10),
    place_of_service VARCHAR(100),
    hcpcs_code VARCHAR(50),
    hcpcs_description TEXT,
    hcpcs_drug_indicator VARCHAR(10),
    number_of_services DECIMAL(20,4),
    number_of_medicare_beneficiaries DECIMAL(20,4),
    number_of_distinct_medicare_beneficiary_per_day_services DECIMAL(20,4),
    average_medicare_allowed_amount DECIMAL(20,6),
    average_submitted_charge_amount DECIMAL(20,6),
    average_medicare_payment_amount DECIMAL(20,6),
    average_medicare_standardized_amount DECIMAL(20,6)
);

-- Provider performance
SELECT provider_type,
       SUM(number_of_services) AS total_services,
       SUM(number_of_medicare_beneficiaries) AS total_beneficiaries,
       AVG(average_medicare_payment_amount) AS avg_payment
FROM healthcare_providers
GROUP BY provider_type
ORDER BY total_services DESC;

-- Geographic utilization
SELECT state_code,
       SUM(number_of_services) AS total_services,
       SUM(number_of_medicare_beneficiaries) AS total_beneficiaries
FROM healthcare_providers
GROUP BY state_code
ORDER BY total_services DESC;

-- Top services
SELECT hcpcs_code, hcpcs_description,
       SUM(number_of_services) AS total_services,
       AVG(average_medicare_payment_amount) AS avg_payment
FROM healthcare_providers
GROUP BY hcpcs_code, hcpcs_description
ORDER BY total_services DESC
LIMIT 10;

-- Cost comparison
SELECT hcpcs_code, hcpcs_description,
       AVG(average_submitted_charge_amount) AS avg_submitted_charge,
       AVG(average_medicare_allowed_amount) AS avg_allowed,
       AVG(average_medicare_payment_amount) AS avg_payment
FROM healthcare_providers
GROUP BY hcpcs_code, hcpcs_description
ORDER BY avg_payment DESC
LIMIT 20;
