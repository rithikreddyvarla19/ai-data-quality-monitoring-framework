create table if not exists monitoring_reports (
  id bigserial primary key,
  run_id text not null,
  dataset_name text not null,
  report_type text not null,
  status text not null,
  score numeric(8, 6) not null,
  payload jsonb not null,
  created_at timestamptz not null default now(),
  unique (run_id, report_type)
);

create index if not exists idx_monitoring_reports_dataset_created
  on monitoring_reports (dataset_name, created_at desc);

create index if not exists idx_monitoring_reports_report_type
  on monitoring_reports (report_type);

create table if not exists monitoring_alerts (
  id bigserial primary key,
  alert_id text not null unique,
  run_id text not null,
  dataset_name text not null,
  severity text not null,
  title text not null,
  message text not null,
  payload jsonb not null,
  created_at timestamptz not null default now()
);

