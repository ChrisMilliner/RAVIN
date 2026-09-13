"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import ravinIcon from "../app/ravin_icon.png";

const links = [
  { href: "/", label: "Home" },
  { href: "/about", label: "About" },
];

export default function SiteHeader() {
  const pathname = usePathname();

  return (
    <header className="site-header">
      <nav className="navbar container" aria-label="Main navigation">
        <Link className="brand" href="/" aria-label="RAVIN home">
          <span className="brand-mark" aria-hidden="true">
            <Image src={ravinIcon} alt="" width={28} height={28} priority />
          </span>
          <span className="brand-text">RAVIN</span>
        </Link>
        <div className="nav-links">
          {links.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              aria-current={pathname === link.href ? "page" : undefined}
            >
              {link.label}
            </Link>
          ))}
        </div>
      </nav>
    </header>
  );
}
