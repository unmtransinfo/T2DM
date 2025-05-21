-- creating temporary table for persons having at least 2 years of observation
\echo '* creating temporary table for persons having at least 2 years of observation'
CREATE temporary table observed_person_table as ( 
	select DISTINCT person_id, min_observation_start_date
	from (
		select person_id, min(observation_period_start_date) as min_observation_start_date, max(observation_period_end_date) as max_observation_end_date 
		from ccae2003_2021.observation_period 
		where observation_period_start_date >= '2018-01-01' 
		group by person_id) f
	where max_observation_end_date - min_observation_start_date >=2*365
);
\copy observed_person_table to PROGRAM 'gzip > /home/pkumar/Diabetes/SQL/Output/diabetes_person_observation_data.tsv.gz' WITH DELIMITER E'\t' CSV HEADER QUOTE E'\b';


-- create a tempory table with all concepts for diabetes
\echo '* creating temporary table for all diabetes concept ids'
CREATE temporary table diabetes_concepts_table as ( 
	select DISTINCT concept_id 
	from omop_20220331.concept 
	where concept_code like 'E11.%' and vocabulary_id='ICD10CM'
);
\copy diabetes_concepts_table to PROGRAM 'gzip > /home/pkumar/Diabetes/SQL/Output/diabetes_concept_codes.tsv.gz' WITH DELIMITER E'\t' CSV HEADER QUOTE E'\b';


-- creating temporary table for details of all observed persons
\echo '* creating temporary table for details of all observed persons'
CREATE temporary table person_details as ( 
	select DISTINCT a.person_id, a.gender_concept_id, a.year_of_birth, a.location_id, b.state 
	from ccae2003_2021.person a, ccae2003_2021.location b 
	where a.location_id=b.location_id
	and a.person_id in (select person_id from observed_person_table)
);
\copy person_details to PROGRAM 'gzip > /home/pkumar/Diabetes/SQL/Output/diabetes_person_data.tsv.gz' WITH DELIMITER E'\t' CSV HEADER QUOTE E'\b';


---select condtions for each person with observed condition
\echo '* selecting conditions from tables'
CREATE temporary table person_conditions as (
	SELECT person_id, condition_concept_id, condition_source_concept_id, condition_start_date
	FROM ccae2003_2021.condition_occurrence
	WHERE person_id IN (SELECT person_id FROM observed_person_table)
);
\copy person_conditions to PROGRAM 'gzip > /home/pkumar/Diabetes/SQL/Output/diabetes_person_conditions_data.tsv.gz' WITH DELIMITER E'\t' CSV HEADER QUOTE E'\b';


---select drugs for each person with observed condition
\echo '* selecting drugs from tables'
CREATE temporary table person_drugs as (
	SELECT person_id, drug_concept_id, drug_era_start_date, drug_era_end_date, drug_exposure_count
	FROM ccae2003_2021.drug_era
	WHERE person_id IN (SELECT person_id FROM observed_person_table)
);
\copy person_drugs to PROGRAM 'gzip > /home/pkumar/Diabetes/SQL/Output/diabetes_person_drugs_data.tsv.gz' WITH DELIMITER E'\t' CSV HEADER QUOTE E'\b';


