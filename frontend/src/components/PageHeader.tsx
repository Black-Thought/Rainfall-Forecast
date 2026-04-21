type Props = {
  title: string;
  description: string;
};

export function PageHeader({ title, description }: Props) {
  return (
    <header className="page-header">
      <h1>{title}</h1>
      <p>{description}</p>
    </header>
  );
}
