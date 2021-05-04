<template>
  <div>
    <h1
      class="text-5xl font-semibold leading-tight text-gray-800 md:text-6xl lg:text-7xl lg:font-medium xl:text-8xl"
    >
      Work Journal
    </h1>
    <div class="mt-8 md:mt-10 xl:mt-16"></div>
    <h2 class="mt-6 text-lg text-gray-700 md:text-xl lg:text-2xl">
      Learning journey and work stuff. Updated weekly.
    </h2>
    <div class="mt-8 md:mt-10 xl:mt-16"></div>
    <nuxt-content
      class="prose sm:prose-sm lg:prose-lg max-w-none"
      :document="page"
    />
  </div>
</template>

<script>
export default {
  async asyncData({ $content, error }) {
    const page = await $content('work-journal')
      .fetch()
      .catch(() => {
        error({ statusCode: 404, message: 'Page not found' })
      })

    return {
      page,
    }
  },
  head() {
    return {
      title: 'Work journal',
      link: [
        {
          rel: 'canonical',
          href: `https://hmatalonga.com${this.$route.path}`,
        },
      ],
      meta: [
        {
          hid: 'og:url',
          property: 'og:url',
          content: `https://hmatalonga.com${this.$route.path}`,
        },
      ],
    }
  },
}
</script>
