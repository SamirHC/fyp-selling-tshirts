CREATE TABLE IF NOT EXISTS "clothes_sources" (
  "id" INTEGER,
  "name" TEXT UNIQUE ON CONFLICT IGNORE,
  PRIMARY KEY ("id")
);

CREATE TABLE IF NOT EXISTS "clothes" (
  "id" INTEGER,
  "source_id" INTEGER,
  "item_id" TEXT,
  "title" TEXT,
  "image_url" TEXT,
  PRIMARY KEY ("id", "source_id"),
  FOREIGN KEY ("source_id") REFERENCES "clothes_sources" ("id")
);

DROP TABLE IF EXISTS "palettes";

CREATE TABLE IF NOT EXISTS "palettes" (
  "id" INTEGER,
  "likes" INTEGER,
  "submission_date" TEXT,
  "color_hunt_id" TEXT UNIQUE ON CONFLICT REPLACE,
  PRIMARY KEY ("id")
);

DROP TABLE IF EXISTS "palette_tags";

CREATE TABLE IF NOT EXISTS "palette_tags" (
  "name" TEXT,
  "is_colour_tag" BOOLEAN,
  PRIMARY KEY ("name")
);

DROP TABLE IF EXISTS "palette_tag_associations";

CREATE TABLE IF NOT EXISTS "palette_tag_associations" (
  "palette_id" INTEGER,
  "tag" TEXT,
  PRIMARY KEY ("palette_id", "tag"),
  FOREIGN KEY ("tag") REFERENCES "palette_tags" ("name")
  FOREIGN KEY ("palette_id") REFERENCES "palettes" ("id")
);

DROP TABLE IF EXISTS "palette_colours";

CREATE TABLE IF NOT EXISTS "palette_colours" (
  "palette_id" INTEGER,
  "colour" CHAR(7),
  PRIMARY KEY ("palette_id", "colour")
  FOREIGN KEY ("palette_id") REFERENCES "palettes" ("id")
);

--DROP TABLE IF EXISTS "print_design_palettes";

CREATE TABLE IF NOT EXISTS "print_design_palettes" (
  "id" INTEGER,
  "source_id" INTEGER,
  "colour" CHAR(7),
  PRIMARY KEY ("id", "source_id", "colour")
  FOREIGN KEY ("id") REFERENCES "clothes" ("id")
  FOREIGN KEY ("source_id") REFERENCES "clothes" ("source_id")
);

CREATE TABLE IF NOT EXISTS "print_design_regions" (
  "id" INTEGER,
  "source_id" INTEGER,
  "left" INTEGER,
  "top" INTEGER,
  "width" INTEGER,
  "height" INTEGER,
  PRIMARY KEY ("id", "source_id")
  FOREIGN KEY ("id") REFERENCES "clothes" ("id")
  FOREIGN KEY ("source_id") REFERENCES "clothes" ("source_id")
);

CREATE TABLE IF NOT EXISTS "palette_distances" (
  "palette_id" INTEGER,
  "design_id" INTEGER,
  "source_id" INTEGER,
  "distance" FLOAT,
  PRIMARY KEY ("palette_id", "design_id", "source_id")
  FOREIGN KEY ("palette_id") REFERENCES "palettes" ("id")
  FOREIGN KEY ("design_id") REFERENCES "clothes" ("id")
  FOREIGN KEY ("source_id") REFERENCES "clothes" ("source_id")
);
