export type PromptIssueType =
  | "ambiguity"
  | "weak_action"
  | "redundancy"
  | "too_short"
  | "missing_output_format"
  | "missing_context"
  | "missing_constraints"
  | "meta_prompt_drift"
  | "answering_risk";

export type PromptIssueSeverity = "low" | "medium" | "high";

export type PromptIssue = {
  id: string;
  type: PromptIssueType;
  severity: PromptIssueSeverity;
  span: string | null;
  start: number | null;
  end: number | null;
  message: string;
  suggestion: string;
};

export type RewriteMetadata = {
  archetype?: string;
  modularity?: string;
  adapter_safe_mode?: boolean;
  runtime_generation_policy?: string;
};

export type ValidationIssue = {
  type: string;
  severity: PromptIssueSeverity;
  message: string;
};

export type ValidationResult = {
  status: "valid" | "invalid";
  issues: ValidationIssue[];
};
