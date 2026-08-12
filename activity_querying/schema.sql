--
-- PostgreSQL database dump
--

-- Dumped from database version 17.2 (Homebrew)
-- Dumped by pg_dump version 17.2 (Homebrew)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: activity; Type: TABLE; Schema: public; Owner: bencrouse
--

CREATE TABLE public.activity (
    id uuid NOT NULL,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    action_type text NOT NULL,
    source character varying(255) NOT NULL,
    tags jsonb DEFAULT '[]'::jsonb NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
    snippet text,
    artifact_type character varying(255) NOT NULL,
    actor_id uuid NOT NULL,
    artifact_id uuid NOT NULL,
    organization_id uuid NOT NULL
);


ALTER TABLE public.activity OWNER TO bencrouse;

--
-- Name: COLUMN activity.source; Type: COMMENT; Schema: public; Owner: bencrouse
--

COMMENT ON COLUMN public.activity.source IS 'INTERNAL: internal
GITHUB: github
GOOGLE_DRIVE: google_drive
MICROSOFT_ONE_DRIVE: microsoft_one_drive
SLACK: slack
WEBSITE: website
SEARCH_ENGINE: search_engine
EMAIL: email';


--
-- Name: COLUMN activity.artifact_type; Type: COMMENT; Schema: public; Owner: bencrouse
--

COMMENT ON COLUMN public.activity.artifact_type IS 'GITHUB_ISSUE: github_issue
GITHUB_PULL_REQUEST: github_pull_request
GITHUB_DISCUSSION: github_discussion
GOOGLE_DOC: google_doc
MEETING: meeting';


--
-- Name: artifact; Type: TABLE; Schema: public; Owner: bencrouse
--

CREATE TABLE public.artifact (
    id uuid NOT NULL,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    type character varying(255) NOT NULL,
    title text NOT NULL,
    url text NOT NULL,
    tags jsonb DEFAULT '[]'::jsonb NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
    organization_id uuid NOT NULL
);


ALTER TABLE public.artifact OWNER TO bencrouse;

--
-- Name: COLUMN artifact.type; Type: COMMENT; Schema: public; Owner: bencrouse
--

COMMENT ON COLUMN public.artifact.type IS 'GITHUB_ISSUE: github_issue
GITHUB_PULL_REQUEST: github_pull_request
GITHUB_DISCUSSION: github_discussion
GOOGLE_DOC: google_doc
MEETING: meeting';


--
-- Name: person; Type: TABLE; Schema: public; Owner: bencrouse
--

CREATE TABLE public.person (
    id uuid NOT NULL,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    name text NOT NULL,
    email text NOT NULL,
    aliases jsonb DEFAULT '[]'::jsonb NOT NULL,
    tags jsonb DEFAULT '[]'::jsonb NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
    organization_id uuid NOT NULL,
    department text
);


ALTER TABLE public.person OWNER TO bencrouse;

--
-- Name: activity activity_pkey; Type: CONSTRAINT; Schema: public; Owner: bencrouse
--

ALTER TABLE ONLY public.activity
    ADD CONSTRAINT activity_pkey PRIMARY KEY (id);


--
-- Name: artifact artifact_pkey; Type: CONSTRAINT; Schema: public; Owner: bencrouse
--

ALTER TABLE ONLY public.artifact
    ADD CONSTRAINT artifact_pkey PRIMARY KEY (id);


--
-- Name: person person_pkey; Type: CONSTRAINT; Schema: public; Owner: bencrouse
--

ALTER TABLE ONLY public.person
    ADD CONSTRAINT person_pkey PRIMARY KEY (id);


--
-- Name: activity activity_actor_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: bencrouse
--

ALTER TABLE ONLY public.activity
    ADD CONSTRAINT activity_actor_id_fkey FOREIGN KEY (actor_id) REFERENCES public.person(id) ON DELETE CASCADE;


--
-- Name: activity activity_artifact_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: bencrouse
--

ALTER TABLE ONLY public.activity
    ADD CONSTRAINT activity_artifact_id_fkey FOREIGN KEY (artifact_id) REFERENCES public.artifact(id) ON DELETE CASCADE;


--
-- Name: activity activity_organization_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: bencrouse
--

ALTER TABLE ONLY public.activity
    ADD CONSTRAINT activity_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES public.organization(id) ON DELETE CASCADE;


--
-- Name: artifact artifact_organization_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: bencrouse
--

ALTER TABLE ONLY public.artifact
    ADD CONSTRAINT artifact_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES public.organization(id) ON DELETE CASCADE;


--
-- Name: person person_organization_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: bencrouse
--

ALTER TABLE ONLY public.person
    ADD CONSTRAINT person_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES public.organization(id) ON DELETE CASCADE;


--
-- Name: TABLE activity; Type: ACL; Schema: public; Owner: bencrouse
--

GRANT ALL ON TABLE public.activity TO decide;


--
-- Name: TABLE artifact; Type: ACL; Schema: public; Owner: bencrouse
--

GRANT ALL ON TABLE public.artifact TO decide;


--
-- Name: TABLE person; Type: ACL; Schema: public; Owner: bencrouse
--

GRANT ALL ON TABLE public.person TO decide;


--
-- PostgreSQL database dump complete
--
